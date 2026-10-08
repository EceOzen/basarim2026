"""
Tek bir config dosyasını okuyup ilgili yöntemi, ilgili domain'de,
n_repeats kadar tekrarla çalıştırır ve sonuçları CSV'ye ekler.

Kullanım:
    python run_experiment.py --config configs/small_siren.yaml

Tüm deney matrisini (3 domain x 3 yöntem) çalıştırmak için configs/
klasöründeki her dosya üzerinde bunu bir döngüyle (bash veya Makefile)
çağırman yeterli — script'in kendisini değiştirmene gerek yok.
"""

import argparse
import csv
import os
import yaml
import numpy as np

from methods.pca import PCAMethod
from methods.siren import SIRENMethod
from methods.hashgrid import HashGridMethod
from utils.timing import Measurement, measure_block
from utils.era5 import load_era5_field, fetch_era5_ensemble_spread
from utils import climatebench


# Registry: config'teki "method" string'ini doğru sınıfa eşliyor.
# Yeni bir yöntem eklersen sadece burada tek satır eklersin.
METHOD_REGISTRY = {
    "pca": PCAMethod,
    "siren": SIRENMethod,
    "hashgrid": HashGridMethod,
}

RESULTS_CSV = "results/experiment_results.csv"
CSV_COLUMNS = [
    "domain", "method", "seed", "repeat_idx",
    "encode_time_s", "decode_time_s",
    "peak_memory_mb", "compression_ratio", "rmse",
    # ClimateBenchPress (Reichelt et al. 2026) metrikleri:
    "max_abs_error", "spectral_error",
    "error_bound_low", "error_bound_mid", "error_bound_high",
    "passes_b_low", "passes_b_mid", "passes_b_high",
]


def load_data(domain_config: dict) -> np.ndarray:
    """
    utils/era5.py -> load_era5_field'i çağırır. domain_config artık
    sadece lat/lon değil, date_range ve resolution da içermeli
    (bkz. configs/*.yaml -- eksik olan bu alanları ekledim).
    """
    region = {
        "lat_min": domain_config["lat_min"],
        "lat_max": domain_config["lat_max"],
        "lon_min": domain_config["lon_min"],
        "lon_max": domain_config["lon_max"],
    }
    date_range = tuple(domain_config["date_range"])
    res = domain_config.get("resolution", 0.25)
    return load_era5_field(region, date_range, res, cache_dir="era5_cache")


def get_error_bounds(domain_config: dict) -> dict:
    """
    ClimateBenchPress error bound'unu domain başına BİR KERE hesaplar
    (tekrar başına değil -- ensemble spread, model seed'inden bağımsız,
    domain'in kendi özelliği). Spread verisi çekilemezse (CDS'te entitlement
    sorunu vb.) pipeline'ı ÇÖKERTMEMEK için NaN bounds ile devam eder --
    run_all.sh'ın "bir config hata verirse devam et" felsefesiyle tutarlı.
    """
    region = {
        "lat_min": domain_config["lat_min"],
        "lat_max": domain_config["lat_max"],
        "lon_min": domain_config["lon_min"],
        "lon_max": domain_config["lon_max"],
    }
    date_range = tuple(domain_config["date_range"])
    res = domain_config.get("resolution", 0.25)
    try:
        spread = fetch_era5_ensemble_spread(region, date_range, res, cache_dir="era5_cache")
        return climatebench.compute_error_bounds(spread)
    except Exception as e:
        print(f"[WARN] Ensemble spread alinamadi ({e}) -- error-bound NaN olarak isaretlenecek")
        return {"b_low": float("nan"), "b_mid": float("nan"), "b_high": float("nan")}


def compute_rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.sqrt(np.mean((original - reconstructed) ** 2)))


def ensure_csv_header():
    """CSV dosyası yoksa oluştur ve başlık satırını yaz."""
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def append_result(row: dict):
    """Tek bir koşunun sonucunu CSV'nin sonuna ekle."""
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


def run(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    domain_name = config["domain"]["name"]
    method_name = config["method"]
    device = config.get("device", "cpu")
    n_repeats = config.get("n_repeats", 1)

    print(f"[run_experiment] domain={domain_name} method={method_name} "
          f"device={device} n_repeats={n_repeats}")

    data = load_data(config["domain"])
    method_cls = METHOD_REGISTRY[method_name]

    # ClimateBenchPress error bound'u domain basina BIR KERE hesaplaniyor
    # (tekrarlar arasinda degismez, seed'e bagli degil).
    bounds = get_error_bounds(config["domain"])
    print(f"[run_experiment] error bounds: b_low={bounds['b_low']:.4f} "
          f"b_mid={bounds['b_mid']:.4f} b_high={bounds['b_high']:.4f}")

    ensure_csv_header()

    for repeat_idx in range(n_repeats):
        # Her tekrarda farklı bir seed kullanıyoruz ki
        # "5-10 tekrar, ortalama ± std" dediğimiz istatistiksel
        # güvenilirliği gerçekten sağlayalım.
        seed = config["hyperparams"].get("seed", 0) + repeat_idx
        hyperparams = {**config["hyperparams"], "seed": seed}

        model = method_cls(**hyperparams)

        encode_result = Measurement()
        with measure_block(encode_result, device=device):
            model.fit(data)

        decode_result = Measurement()
        with measure_block(decode_result, device=device):
            reconstructed = model.reconstruct()

        rmse = compute_rmse(data, reconstructed)
        bound_eval = climatebench.evaluate_against_bounds(rmse, bounds)

        row = {
            "domain": domain_name,
            "method": method_name,
            "seed": seed,
            "repeat_idx": repeat_idx,
            "encode_time_s": encode_result.elapsed_seconds,
            "decode_time_s": decode_result.elapsed_seconds,
            "peak_memory_mb": encode_result.peak_memory_mb,  # genelde encode daha yüklü
            "compression_ratio": model.compression_ratio(data),
            "rmse": rmse,
            "max_abs_error": climatebench.max_abs_error(data, reconstructed),
            "spectral_error": climatebench.spectral_error(data, reconstructed),
            "error_bound_low": bounds["b_low"],
            "error_bound_mid": bounds["b_mid"],
            "error_bound_high": bounds["b_high"],
            "passes_b_low": bound_eval["passes_b_low"],
            "passes_b_mid": bound_eval["passes_b_mid"],
            "passes_b_high": bound_eval["passes_b_high"],
        }
        append_result(row)
        print(f"  repeat {repeat_idx}: "
              f"encode={row['encode_time_s']:.3f}s "
              f"decode={row['decode_time_s']:.3f}s "
              f"rmse={row['rmse']:.4f} "
              f"passes_b_mid={row['passes_b_mid']}")


    print(f"[run_experiment] Tamamlandı. Sonuçlar: {RESULTS_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="config yaml dosyasının yolu")
    args = parser.parse_args()
    run(args.config)
