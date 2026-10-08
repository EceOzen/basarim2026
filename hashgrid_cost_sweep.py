"""
Hash-grid per-step cost sweep (BAŞARIM camera-ready, Section VI-B).

Amaç: Makaledeki "hash-grid adım başı maliyeti L ile doğrusal, T'den bağımsız"
iddiasını değişkenleri İZOLE ederek doğrulamak. Ana deneyde domain, L, T ve
N_max aynı anda değişiyordu; burada her seferinde tek bir değişkeni oynatıyoruz.

methods/hashgrid.py'ye DOKUNMAZ -- makaledeki implementasyonun kendisini ölçer.

Ölçüm yöntemi (sabit maliyetleri ayıklamak için iki-nokta farkı):
    t(E) = setup + E * step       (setup = meshgrid, GPU'ya kopya, model kurulumu)
    step = (t(E_long) - t(E_short)) / (E_long - E_short)
Her konfigürasyon için bir kez ısınma (warm-up) koşusu yapılır, sonra
--repeats kadar tekrar edilir ve medyan raporlanır.

Veri: Adım başı maliyet veri DEĞERLERİNDEN bağımsızdır (batch boyutu sabit,
indeksler rastgele), sadece şekil önemlidir. Varsayılan olarak small domain
şeklinde (9 x 17 x 720) sentetik veri kullanılır. Gerçek veriyle koşmak
istersen: --data path/to/small.npy  (şekil: n_lat, n_lon, n_time)

Kullanım (GPU makinede):
    cd experiments
    python hashgrid_cost_sweep.py                  # tüm taramalar + profiler + SIREN referansı
    python hashgrid_cost_sweep.py --quick          # hızlı duman testi (~1 dk)
    python hashgrid_cost_sweep.py --sweeps L T     # sadece bazı taramalar

Çıktılar (results/cost_sweep/):
    sweep_raw.csv       her ölçüm satırı
    sweep_summary.csv   konfigürasyon başına medyan ms/adım
    profile.csv         adım başına CUDA kernel sayısı (L'ye göre)
    cost_sweep.png/pdf  makale için şekil
    summary.txt         doğrusal fit (ms/seviye, kesişim, R^2) ve özet
"""

import argparse
import csv
import os
import platform
import statistics
import time

import numpy as np
import torch

from methods.hashgrid import HashGridMethod
from methods.siren import SIRENMethod

# Makaledeki small-domain hash-grid konfigürasyonu (Tablo II) -- taramaların "taban" noktası
BASE = dict(
    n_levels=8,
    n_features_per_level=2,
    log2_hashmap_size=9,
    base_resolution=4,
    finest_resolution=32,
    mlp_hidden_dim=64,
    batch_size=16384,
)

SWEEPS = {
    # Sadece L değişiyor (T=9, N_max=32 sabit). Beklenti: doğrusal artış.
    "L": [("n_levels", v) for v in [2, 4, 6, 8, 10, 12, 14, 16]],
    # Sadece T değişiyor (L=8, N_max=32 sabit). Beklenti: düz.
    "T": [("log2_hashmap_size", v) for v in [9, 11, 13, 14, 15, 17, 19]],
    # Sadece N_max değişiyor (L=8, T=9 sabit). Beklenti: düz.
    "N": [("finest_resolution", v) for v in [16, 32, 64, 128, 256]],
    # Ana deneydeki üç gerçek konfigürasyon (Tablo II), aynı veri şekliyle --
    # Tablo III'ten türetilen 23.5 / 29.3 / 34.5 ms/adım ile karşılaştırma için.
    "paper": [
        ("paper", dict(n_levels=8, log2_hashmap_size=9, finest_resolution=32)),
        ("paper", dict(n_levels=10, log2_hashmap_size=13, finest_resolution=128)),
        ("paper", dict(n_levels=12, log2_hashmap_size=14, finest_resolution=128)),
    ],
}

PAPER_MS_PER_STEP = {8: 23.5, 10: 29.3, 12: 34.5}  # Tablo III encode süresi / epoch


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_fit(method_cls, data, hp):
    sync()
    t0 = time.perf_counter()
    m = method_cls(**hp)
    m.fit(data)
    sync()
    return time.perf_counter() - t0, m


def per_step_ms(method_cls, data, hp, e_short, e_long):
    """İki farklı epoch sayısıyla koşup farktan saf adım başı süreyi çıkarır."""
    t_short, _ = timed_fit(method_cls, data, {**hp, "n_epochs": e_short})
    t_long, _ = timed_fit(method_cls, data, {**hp, "n_epochs": e_long})
    step = (t_long - t_short) / (e_long - e_short)
    setup = t_short - e_short * step
    return step * 1e3, setup


def count_kernels_per_step(data, hp, n_steps=20):
    """torch.profiler ile adım başına CUDA kernel sayısı ve GPU-aktif süre oranı."""
    from torch.profiler import profile, ProfilerActivity

    if not torch.cuda.is_available():
        return None
    # ısınma
    timed_fit(HashGridMethod, data, {**hp, "n_epochs": 5})

    def run(n):
        timed_fit(HashGridMethod, data, {**hp, "n_epochs": n})

    # setup kernel'larını ayıklamak için iki uzunlukta profil alıp farkı kullanıyoruz
    results = {}
    for n in (n_steps, 2 * n_steps):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            sync()
            t0 = time.perf_counter()
            run(n)
            sync()
            wall = time.perf_counter() - t0
        events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        n_kernels = len(events)
        gpu_us = sum(e.device_time for e in events) if events and hasattr(events[0], "device_time") \
            else sum(e.cuda_time for e in events)
        results[n] = (n_kernels, gpu_us, wall)

    (k1, g1, w1), (k2, g2, w2) = results[n_steps], results[2 * n_steps]
    kernels_per_step = (k2 - k1) / n_steps
    gpu_ms_per_step = (g2 - g1) / n_steps / 1e3
    wall_ms_per_step = (w2 - w1) / n_steps * 1e3
    return dict(
        kernels_per_step=kernels_per_step,
        gpu_busy_ms_per_step=gpu_ms_per_step,
        wall_ms_per_step=wall_ms_per_step,
        gpu_busy_fraction=gpu_ms_per_step / wall_ms_per_step if wall_ms_per_step > 0 else float("nan"),
    )


def linfit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, r2


def make_plot(summary, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("L", "Number of levels $L$", "n_levels"),
              ("T", r"$\log_2$ hash table size $T$", "log2_hashmap_size"),
              ("N", r"Finest resolution $N_{\max}$", "finest_resolution")]
    panels = [p for p in panels if any(r["sweep"] == p[0] for r in summary)]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    siren = [r for r in summary if r["sweep"] == "siren"]

    for ax, (sw, label, key) in zip(axes, panels):
        rows = sorted([r for r in summary if r["sweep"] == sw], key=lambda r: r[key])
        x = [r[key] for r in rows]
        y = [r["ms_per_step_median"] for r in rows]
        lo = [r["ms_per_step_median"] - r["ms_per_step_min"] for r in rows]
        hi = [r["ms_per_step_max"] - r["ms_per_step_median"] for r in rows]
        ax.errorbar(x, y, yerr=[lo, hi], fmt="o-", color="#2a6f97", ms=4, lw=1.4, capsize=2,
                    label="Hash-grid")
        if sw == "L" and len(x) >= 2:
            s, b, r2 = linfit(x, y)
            xx = np.linspace(min(x), max(x), 50)
            ax.plot(xx, s * xx + b, "--", color="#999999", lw=1,
                    label=f"fit: {s:.2f} ms/level, $R^2$={r2:.3f}")
        if siren:
            ax.axhline(siren[0]["ms_per_step_median"], color="#c1666b", lw=1, ls=":",
                       label="SIREN")
        if sw == "N":
            ax.set_xscale("log", base=2)
        ax.set_xlabel(label)
        ax.grid(alpha=0.3, lw=0.5)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("ms per training step")
    axes[0].set_ylim(bottom=0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"cost_sweep.{ext}"), dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help=".npy dosyası (n_lat, n_lon, n_time). Yoksa sentetik.")
    ap.add_argument("--shape", type=int, nargs=3, default=[9, 17, 720],
                    help="Sentetik veri şekli (varsayılan: small domain 9x17x720)")
    ap.add_argument("--sweeps", nargs="+", default=["L", "T", "N", "paper"],
                    choices=list(SWEEPS.keys()))
    ap.add_argument("--e-short", type=int, default=100)
    ap.add_argument("--e-long", type=int, default=600)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--siren-hidden", type=int, default=96)
    ap.add_argument("--siren-depth", type=int, default=5)
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--no-siren", action="store_true")
    ap.add_argument("--quick", action="store_true", help="Duman testi: az epoch, 1 tekrar, kısa taramalar")
    ap.add_argument("--out", default="results/cost_sweep")
    args = ap.parse_args()

    if args.quick:
        args.e_short, args.e_long, args.repeats = 5, 15, 1
        for k in ("L", "T", "N"):
            SWEEPS[k] = SWEEPS[k][::3]

    os.makedirs(args.out, exist_ok=True)

    if args.data:
        data = np.load(args.data).astype(np.float32)
        data_src = args.data
    else:
        rng = np.random.default_rng(0)
        data = (rng.standard_normal(tuple(args.shape)) * 5 + 15).astype(np.float32)
        data_src = f"synthetic {tuple(args.shape)}"

    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    header = (f"device: {dev} | torch {torch.__version__} | python {platform.python_version()}\n"
              f"data: {data_src} | epochs: {args.e_short}->{args.e_long} | repeats: {args.repeats}\n")
    print(header)
    if not torch.cuda.is_available():
        print("UYARI: CUDA yok -- sayılar makaledeki GPU ölçümleriyle karşılaştırılamaz.\n")

    # Genel ısınma (CUDA context, cuBLAS init vb. ilk ölçümü kirletmesin)
    timed_fit(HashGridMethod, data, {**BASE, "n_epochs": 10, "seed": 0})

    raw, summary = [], []

    def measure(sweep, method_cls, hp, tag):
        timed_fit(method_cls, data, {**hp, "n_epochs": 5, "seed": 0})  # konfigürasyon başına ısınma
        steps = []
        for r in range(args.repeats):
            ms, setup = per_step_ms(method_cls, data, {**hp, "seed": r}, args.e_short, args.e_long)
            steps.append(ms)
            raw.append(dict(sweep=sweep, tag=tag, repeat=r, ms_per_step=round(ms, 4),
                            setup_s=round(setup, 4), **{k: hp.get(k, "") for k in
                            ("n_levels", "log2_hashmap_size", "finest_resolution")}))
        row = dict(sweep=sweep, tag=tag,
                   n_levels=hp.get("n_levels", ""), log2_hashmap_size=hp.get("log2_hashmap_size", ""),
                   finest_resolution=hp.get("finest_resolution", ""),
                   ms_per_step_median=statistics.median(steps),
                   ms_per_step_min=min(steps), ms_per_step_max=max(steps))
        if method_cls is HashGridMethod:
            enc_params = hp["n_levels"] * (2 ** hp["log2_hashmap_size"]) * hp["n_features_per_level"]
            row["ms_per_level"] = row["ms_per_step_median"] / hp["n_levels"]
            row["hash_table_params"] = enc_params
        summary.append(row)
        print(f"  [{sweep:>5}] {tag:<28} {row['ms_per_step_median']:8.3f} ms/step "
              f"(min {row['ms_per_step_min']:.3f}, max {row['ms_per_step_max']:.3f})")

    if not args.no_siren:
        print("SIREN referansı:")
        measure("siren", SIRENMethod,
                dict(hidden_dim=args.siren_hidden, depth=args.siren_depth, w0=5, batch_size=16384),
                f"SIREN h={args.siren_hidden} d={args.siren_depth}")

    for sw in args.sweeps:
        print(f"Tarama {sw}:")
        for key, val in SWEEPS[sw]:
            hp = {**BASE, **val} if key == "paper" else {**BASE, key: val}
            tag = f"L={hp['n_levels']} T={hp['log2_hashmap_size']} Nmax={hp['finest_resolution']}"
            measure(sw, HashGridMethod, hp, tag)

    # CSV'ler
    def write_csv(path, rows):
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    write_csv(os.path.join(args.out, "sweep_raw.csv"), raw)
    write_csv(os.path.join(args.out, "sweep_summary.csv"), summary)

    # Profiler: adım başına kernel sayısı (L'ye göre). Beklenti: ~ a + b*L
    prof_rows = []
    if not args.no_profile and torch.cuda.is_available():
        print("Profiler (adım başına CUDA kernel):")
        for L in ([4, 8, 12] if not args.quick else [8]):
            p = count_kernels_per_step(data, {**BASE, "n_levels": L, "seed": 0})
            if p:
                prof_rows.append(dict(n_levels=L, **{k: round(v, 3) for k, v in p.items()}))
                print(f"  L={L:>2}: {p['kernels_per_step']:.0f} kernel/adım, "
                      f"GPU meşgul {p['gpu_busy_ms_per_step']:.2f} ms / duvar {p['wall_ms_per_step']:.2f} ms "
                      f"(%{100 * p['gpu_busy_fraction']:.0f})")
        if prof_rows:
            write_csv(os.path.join(args.out, "profile.csv"), prof_rows)

    # Özet metni
    lines = [header]
    L_rows = sorted([r for r in summary if r["sweep"] == "L"], key=lambda r: r["n_levels"])
    if len(L_rows) >= 2:
        s, b, r2 = linfit([r["n_levels"] for r in L_rows], [r["ms_per_step_median"] for r in L_rows])
        lines.append(f"L taraması: ms/adım = {s:.3f} * L + {b:.3f}   (R^2 = {r2:.4f})")
        lines.append(f"  -> seviye başına maliyet ≈ {s:.2f} ms; L'den bağımsız sabit kısım ≈ {b:.2f} ms")
    for sw, key in (("T", "log2_hashmap_size"), ("N", "finest_resolution")):
        rows = [r for r in summary if r["sweep"] == sw]
        if rows:
            ys = [r["ms_per_step_median"] for r in rows]
            lines.append(f"{sw} taraması: {min(ys):.2f}–{max(ys):.2f} ms/adım "
                         f"(göreli yayılım %{100 * (max(ys) - min(ys)) / statistics.mean(ys):.1f}) "
                         f"over {key} = {[r[key] for r in rows]}")
    paper_rows = [r for r in summary if r["sweep"] == "paper"]
    if paper_rows:
        lines.append("Makale konfigürasyonları (bu ölçüm vs. Tablo III'ten türetilen):")
        for r in paper_rows:
            ref = PAPER_MS_PER_STEP.get(r["n_levels"])
            lines.append(f"  {r['tag']:<28} {r['ms_per_step_median']:7.2f} ms  vs  {ref} ms")
    sir = [r for r in summary if r["sweep"] == "siren"]
    if sir:
        lines.append(f"SIREN: {sir[0]['ms_per_step_median']:.2f} ms/adım (makale ≈1.8 ms)")
    if prof_rows:
        lines.append("Profiler:")
        for p in prof_rows:
            lines.append(f"  L={p['n_levels']}: {p['kernels_per_step']:.0f} kernel/adım, "
                         f"GPU meşgul oranı %{100 * p['gpu_busy_fraction']:.0f}")
    text = "\n".join(lines)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)

    make_plot(summary, args.out)
    print(f"\nÇıktılar: {args.out}/")


if __name__ == "__main__":
    main()
