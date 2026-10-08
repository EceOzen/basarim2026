"""
Deney sonuçlarını (results/experiment_results.csv) okuyup okunabilir bir
markdown rapor + destekleyici grafikler üretir.

Neden ayrı bir modül: run_experiment.py sadece HAM veriyi topluyor
(her koşu bir satır). Bu dosya o ham veriyi ANLAMLI hale getiriyor --
ortalama ± std, domain'ler arası scaling trendi, ve iki farklı
"anlatı" için (BAŞARIM: HPC metrikleri, REO-2: representation quality)
ayrı tablolar.

Kullanım:
    python -m utils.report
    # ya da özel bir CSV/çıktı klasörü için:
    python -m utils.report --csv results/experiment_results.csv --out results/report
"""

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


# Domain'lerin "büyüklük sırası" -- grafik ve tablolarda bu sırayla göstermek için.
# CSV'deki domain isimleriyle birebir eşleşmeli (configs/*.yaml -> domain.name).
DOMAIN_ORDER = ["small", "medium", "large"]
METHOD_ORDER = ["pca", "siren", "hashgrid"]


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["domain"] = pd.Categorical(df["domain"], categories=DOMAIN_ORDER, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Her (domain, method) kombinasyonunun tekrarlarını ortalama ± std'ye indirger."""
    agg = df.groupby(["domain", "method"], observed=True).agg(
        encode_time_mean=("encode_time_s", "mean"),
        encode_time_std=("encode_time_s", "std"),
        decode_time_mean=("decode_time_s", "mean"),
        decode_time_std=("decode_time_s", "std"),
        peak_memory_mean=("peak_memory_mb", "mean"),
        peak_memory_std=("peak_memory_mb", "std"),
        compression_ratio_mean=("compression_ratio", "mean"),
        compression_ratio_std=("compression_ratio", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        max_abs_error_mean=("max_abs_error", "mean"),
        max_abs_error_std=("max_abs_error", "std"),
        spectral_error_mean=("spectral_error", "mean"),
        spectral_error_std=("spectral_error", "std"),
        passes_b_low_rate=("passes_b_low", "mean"),   # True/False ortalaması = geçme oranı
        passes_b_mid_rate=("passes_b_mid", "mean"),
        passes_b_high_rate=("passes_b_high", "mean"),
        n_runs=("seed", "count"),
    ).reset_index()
    return agg.sort_values(["domain", "method"])


def _fmt(mean, std, unit=""):
    if pd.isna(mean):
        return "—"
    if pd.isna(std):
        return f"{mean:.3f}{unit}"
    return f"{mean:.3f} ± {std:.3f}{unit}"


def _hpc_table_md(summary: pd.DataFrame) -> str:
    """BAŞARIM için: encode/decode süresi ve bellek öne çıkıyor."""
    lines = [
        "| Domain | Yöntem | Encode (s) | Decode (s) | Peak Memory (MB) | Tekrar |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['domain']} | {r['method']} "
            f"| {_fmt(r['encode_time_mean'], r['encode_time_std'])} "
            f"| {_fmt(r['decode_time_mean'], r['decode_time_std'])} "
            f"| {_fmt(r['peak_memory_mean'], r['peak_memory_std'])} "
            f"| {int(r['n_runs'])} |"
        )
    return "\n".join(lines)


def _repr_table_md(summary: pd.DataFrame) -> str:
    """REO-2 için: representation kalitesi (compression ratio, RMSE) öne çıkıyor."""
    lines = [
        "| Domain | Yöntem | Compression Ratio | RMSE (°C) | Tekrar |",
        "|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['domain']} | {r['method']} "
            f"| {_fmt(r['compression_ratio_mean'], r['compression_ratio_std'], 'x')} "
            f"| {_fmt(r['rmse_mean'], r['rmse_std'])} "
            f"| {int(r['n_runs'])} |"
        )
    return "\n".join(lines)


def _climatebench_table_md(summary: pd.DataFrame) -> str:
    """ClimateBenchPress (Reichelt et al. 2026) error-bound karşılaştırması --
    RMSE'yi oznel degil, ERA5'in kendi belirsizliginden turetilmis bir esige
    gore degerlendiriyor. 'Gecti' oranı, o (domain,yontem) kombinasyonunun
    5-10 tekrarindan kaci esigi gectigini gosteriyor."""
    lines = [
        "| Domain | Yöntem | b_low geçti | b_mid geçti | b_high geçti | MaxAbsError | Spectral Error |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['domain']} | {r['method']} "
            f"| {r['passes_b_low_rate']:.0%} "
            f"| {r['passes_b_mid_rate']:.0%} "
            f"| {r['passes_b_high_rate']:.0%} "
            f"| {_fmt(r['max_abs_error_mean'], r['max_abs_error_std'])} "
            f"| {_fmt(r['spectral_error_mean'], r['spectral_error_std'])} |"
        )
    return "\n".join(lines)


def _scaling_table_md(summary: pd.DataFrame) -> str:
    """Domain büyüdükçe her yöntemin encode süresi kaç kat arttı? (BAŞARIM'ın
    'scalability' bölümü için doğrudan kullanılabilir bir özet.)"""
    lines = [
        "| Yöntem | small -> medium (encode süresi katı) | medium -> large (encode süresi katı) |",
        "|---|---|---|",
    ]
    for method in METHOD_ORDER:
        sub = summary[summary["method"] == method].set_index("domain")
        try:
            small_t = sub.loc["small", "encode_time_mean"]
            medium_t = sub.loc["medium", "encode_time_mean"]
            large_t = sub.loc["large", "encode_time_mean"]
            ratio_sm = medium_t / small_t if small_t else float("nan")
            ratio_ml = large_t / medium_t if medium_t else float("nan")
            lines.append(f"| {method} | {ratio_sm:.2f}x | {ratio_ml:.2f}x |")
        except KeyError:
            lines.append(f"| {method} | veri eksik | veri eksik |")
    return "\n".join(lines)


def _make_plots(summary: pd.DataFrame, df: pd.DataFrame, out_dir: str) -> list:
    """Üç grafik üretir: encode süresi scaling, memory scaling, RMSE-vs-compression pareto."""
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    # --- 1) Encode süresi vs domain (BAŞARIM scaling grafiği) ---
    fig, ax = plt.subplots(figsize=(6, 4))
    for method in METHOD_ORDER:
        sub = summary[summary["method"] == method]
        if sub.empty:
            continue
        ax.errorbar(
            sub["domain"].astype(str), sub["encode_time_mean"], yerr=sub["encode_time_std"],
            marker="o", label=method, capsize=3,
        )
    ax.set_xlabel("Domain")
    ax.set_ylabel("Encode süresi (s)")
    ax.set_title("Encode süresi vs. domain boyutu")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, "encode_time_scaling.png")
    plt.savefig(p, dpi=200)
    plt.close(fig)
    saved.append(p)

    # --- 2) Peak memory vs domain ---
    fig, ax = plt.subplots(figsize=(6, 4))
    for method in METHOD_ORDER:
        sub = summary[summary["method"] == method]
        if sub.empty:
            continue
        ax.errorbar(
            sub["domain"].astype(str), sub["peak_memory_mean"], yerr=sub["peak_memory_std"],
            marker="s", label=method, capsize=3,
        )
    ax.set_xlabel("Domain")
    ax.set_ylabel("Peak GPU memory (MB)")
    ax.set_title("Peak memory vs. domain boyutu")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, "memory_scaling.png")
    plt.savefig(p, dpi=200)
    plt.close(fig)
    saved.append(p)

    # --- 3) RMSE vs compression ratio (REO-2 pareto grafiği) ---
    fig, ax = plt.subplots(figsize=(6, 4))
    markers = {"pca": "s", "siren": "o", "hashgrid": "^"}
    for method in METHOD_ORDER:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        ax.scatter(sub["compression_ratio"], sub["rmse"], label=method,
                   marker=markers.get(method, "o"), alpha=0.7)
    ax.set_xlabel("Compression ratio")
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("RMSE vs. compression ratio (tüm domain/tekrarlar)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, "rmse_vs_compression_pareto.png")
    plt.savefig(p, dpi=200)
    plt.close(fig)
    saved.append(p)

    return saved


def generate_report(csv_path: str = "results/experiment_results.csv",
                     out_dir: str = "results/report") -> str:
    """
    Ana fonksiyon. CSV'yi okur, özetler, markdown rapor + grafikleri
    out_dir altına yazar. Rapor dosyasının yolunu döndürür.
    """
    df = load_results(csv_path)
    summary = summarize(df)

    os.makedirs(out_dir, exist_ok=True)
    plot_paths = _make_plots(summary, df, out_dir)

    report_md = f"""# Deney Sonuçları Raporu

Kaynak: `{csv_path}`
Toplam koşu sayısı: {len(df)}

## 1. HPC Metrikleri (BAŞARIM için)

{_hpc_table_md(summary)}

![Encode süresi scaling]({os.path.basename(plot_paths[0])})
![Memory scaling]({os.path.basename(plot_paths[1])})

## 2. Representation Kalitesi (REO-2 için)

{_repr_table_md(summary)}

![RMSE vs compression pareto]({os.path.basename(plot_paths[2])})

## 3. ClimateBenchPress Error-Bound Değerlendirmesi (Reichelt et al. 2026)

RMSE'nin ERA5'in kendi ensemble spread'inden türetilen fiziksel hata sınırlarına
göre değerlendirilmesi. "Geçti" oranı, o kombinasyonun tekrarlarından kaçının
ilgili sınırı sağladığını gösterir (örn. %100 = tüm tekrarlar geçti, %0 = hiçbiri geçmedi).

{_climatebench_table_md(summary)}

## 4. Scalability Özeti

Domain büyüdükçe encode süresinin kaç kat arttığı (strong/weak scaling
tartışmasında doğrudan kullanılabilir):

{_scaling_table_md(summary)}

## 5. Ham Özet Tablosu

{summary.to_markdown(index=False)}
"""

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report_md)

    # Özet CSV'yi de ayrıca kaydediyoruz -- makale yazarken tabloyu
    # tekrar hesaplamak yerine doğrudan buradan kopyalayabilirsin.
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    print(f"[report] Rapor yazıldı: {report_path}")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/experiment_results.csv")
    parser.add_argument("--out", default="results/report")
    args = parser.parse_args()
    generate_report(args.csv, args.out)
