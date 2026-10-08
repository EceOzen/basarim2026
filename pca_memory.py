"""
PCA peak memory + decode süresi (ms) + alan istatistikleri (BAŞARIM camera-ready, Tablo III / V).

Hakem 3: "PCA decode süresi 0.000 ± 0.000 s yanıltıcı" ve "PCA peak memory neden yok?"
Bu script ana deneydeki PCAMethod'u ve load_data'yı DEĞİŞTİRMEDEN kullanır.

Ölçülenler (domain başına, ısınma koşusundan SONRA, n_repeats tekrar):
  - encode / decode süresi (ms, medyan ve ortalama ± std)
  - peak memory, iki yöntemle:
      * tracemalloc: Python + NumPy'nin fit/reconstruct sırasında ayırdığı tepe bellek
        (NumPy tahsisleri tracemalloc'a raporlanır). Girdi verisi HARİÇ -- sadece
        yöntemin kendi ek belleği. GPU'daki max_memory_allocated ile en yakın kavram.
      * RSS artışı: süreç bellek kullanımındaki tepe artış (psutil ile örneklenir).
        BLAS/LAPACK'in C tarafındaki tamponlarını da yakalar.
  - alan std / ortalama (normalize RMSE için: RMSE / std)

GPU taramasıyla AYNI ANDA çalıştırma -- CPU'yu meşgul eder, kernel launch ölçümlerini bozar.

Kullanım (experiments/ içinde, run_experiment.py ile aynı yerde):
    python pca_memory.py
    python pca_memory.py --configs configs/small_pca.yaml   # tek domain
Çıktı: results/pca_memory.csv ve ekrana özet tablo.
"""

import argparse
import csv
import gc
import os
import statistics
import threading
import time
import tracemalloc

import numpy as np
import yaml

from methods.pca import PCAMethod
from run_experiment import load_data

try:
    import psutil
except ImportError:
    psutil = None


class RSSPeak:
    """Arka planda RSS'i örnekleyip blok süresince tepe artışı ölçer."""

    def __init__(self, interval=0.0005):
        self.interval = interval
        self.peak_delta_mb = float("nan")

    def __enter__(self):
        if psutil is None:
            return self
        self._proc = psutil.Process()
        self._base = self._proc.memory_info().rss
        self._peak = self._base
        self._stop = threading.Event()

        def sample():
            while not self._stop.is_set():
                self._peak = max(self._peak, self._proc.memory_info().rss)
                time.sleep(self.interval)

        self._t = threading.Thread(target=sample, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        if psutil is None:
            return
        self._peak = max(self._peak, self._proc.memory_info().rss)
        self._stop.set()
        self._t.join()
        self.peak_delta_mb = (self._peak - self._base) / 2**20


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t0) * 1e3, out


def measure_once(data, hp):
    gc.collect()
    m = PCAMethod(**hp)

    # 1) süre (bellek izleme KAPALI -- tracemalloc yavaşlatır)
    enc_ms, _ = timed(lambda: m.fit(data))
    dec_ms, recon = timed(m.reconstruct)
    rmse = float(np.sqrt(np.mean((data - recon) ** 2)))
    del recon

    # 2) bellek (ayrı geçiş, süre ölçümünü kirletmesin)
    gc.collect()
    m2 = PCAMethod(**hp)
    tracemalloc.start()
    with RSSPeak() as rss_enc:
        m2.fit(data)
    _, enc_trace_peak = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    with RSSPeak() as rss_dec:
        recon2 = m2.reconstruct()
    _, dec_trace_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del recon2

    return dict(
        encode_ms=enc_ms,
        decode_ms=dec_ms,
        rmse=rmse,
        encode_peak_tracemalloc_mb=enc_trace_peak / 2**20,
        decode_peak_tracemalloc_mb=dec_trace_peak / 2**20,
        encode_peak_rss_delta_mb=rss_enc.peak_delta_mb,
        decode_peak_rss_delta_mb=rss_dec.peak_delta_mb,
        compression_ratio=m.compression_ratio(data),
    )


def fmt(xs, nd=3):
    if len(xs) > 1:
        return f"{statistics.mean(xs):.{nd}f} ± {statistics.stdev(xs):.{nd}f} (med {statistics.median(xs):.{nd}f})"
    return f"{xs[0]:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+",
                    default=["configs/small_pca.yaml", "configs/medium_pca.yaml", "configs/large_pca.yaml"])
    ap.add_argument("--repeats", type=int, default=None, help="Varsayılan: config'teki n_repeats")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="results/pca_memory.csv")
    args = ap.parse_args()

    if psutil is None:
        print("UYARI: psutil yok (pip install psutil) -- sadece tracemalloc raporlanacak.\n")

    rows, summary_lines = [], []
    for cfg_path in args.configs:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        name = cfg["domain"]["name"]
        data = load_data(cfg["domain"]).astype(np.float32, copy=False)
        n_rep = args.repeats or cfg.get("n_repeats", 5)
        hp = dict(cfg["hyperparams"])

        field_std = float(data.std())
        field_mean = float(data.mean())
        data_mb = data.nbytes / 2**20
        print(f"[{name}] shape={data.shape} data={data_mb:.2f} MB  mean={field_mean:.3f}  std={field_std:.3f}")

        for _ in range(args.warmup):  # ısınma: BLAS thread havuzu, sklearn import yolları, sayfa hataları
            PCAMethod(**hp).fit(data)

        res = []
        for r in range(n_rep):
            out = measure_once(data, hp)
            out.update(domain=name, repeat=r, shape="x".join(map(str, data.shape)),
                       data_mb=data_mb, field_mean=field_mean, field_std=field_std,
                       nrmse=out["rmse"] / field_std)
            res.append(out)
            rows.append(out)

        col = lambda k: [x[k] for x in res]
        summary_lines += [
            f"[{name}]  data {data_mb:.2f} MB, field std {field_std:.3f} K",
            f"  encode ms          : {fmt(col('encode_ms'))}",
            f"  decode ms          : {fmt(col('decode_ms'))}",
            f"  peak tracemalloc MB: encode {fmt(col('encode_peak_tracemalloc_mb'), 2)} | decode {fmt(col('decode_peak_tracemalloc_mb'), 2)}",
            f"  peak RSS delta MB  : encode {fmt(col('encode_peak_rss_delta_mb'), 2)} | decode {fmt(col('decode_peak_rss_delta_mb'), 2)}",
            f"  RMSE {fmt(col('rmse'), 4)}  ->  nRMSE (RMSE/std) {fmt(col('nrmse'), 4)}",
            "",
        ]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    text = "\n".join(summary_lines)
    with open(os.path.splitext(args.out)[0] + "_summary.txt", "w") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"Çıktılar: {args.out} ve {os.path.splitext(args.out)[0]}_summary.txt")


if __name__ == "__main__":
    main()
