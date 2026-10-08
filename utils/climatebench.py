"""
ClimateBenchPress (Reichelt et al. 2026, GMD) metodolojisi -- daha once
notebook'ta gelistirdigin error-bound + MaxAbsError + Spectral Error (RAPSD)
hesaplamalarinin, deney pipeline'ina tasinmis hali.

MANTIK:
RMSE tek basina "iyi mi kotu mu" sorusuna oznel bir cevap verir. Bu modul,
ERA5'in kendi ensemble spread'inden (belirsizlik tahmini) FIZIKSEL olarak
temellendirilmis bir hata esigi turetir -- boylece "RMSE 0.384 iyi mi?"
yerine "RMSE, ERA5'in kendi belirsizliginin X katı" diyebiliyorsun.

Ayrica RMSE'nin gizleyebilecegi iki basarisizlik modunu de yakaliyoruz:
  - MaxAbsError: tek bir noktada patlama var mi (RMSE ortalama oldugu icin gizleyebilir)
  - Spectral Error (RAPSD): yuksek frekansli (ince detay) kayip var mi
    (RMSE dusuk olsa bile goruntu "bulanik" olabilir, RAPSD bunu yakalar)
"""

import numpy as np


def compute_error_bounds(spread: np.ndarray) -> dict:
    """
    ERA5 ensemble spread verisinden 3 seviyeli hata esigi turetir
    (notebook'taki b_low/b_mid/b_high mantigiyla birebir ayni: percentile 0/1/5).

    b_low  (strict):  en sıkı esik -- spread'in minimumu
    b_mid  (medium):  orta esik -- spread'in 1. persentili
    b_high (lenient): en gevsek esik -- spread'in 5. persentili
    """
    nonzero_spread = spread[spread > 0].ravel()
    return {
        "b_low": float(np.percentile(nonzero_spread, 0)),
        "b_mid": float(np.percentile(nonzero_spread, 1)),
        "b_high": float(np.percentile(nonzero_spread, 5)),
    }


def max_abs_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.max(np.abs(original - reconstructed)))


def radially_averaged_psd(field_2d: np.ndarray) -> np.ndarray:
    """2D bir alanin radially-averaged power spectral density'si (pysteps konvansiyonu)."""
    F = np.fft.fftshift(np.fft.fft2(field_2d))
    psd2d = (np.abs(F) ** 2) / field_2d.size
    ny, nx = field_2d.shape
    cy, cx = ny // 2, nx // 2
    y, x = np.indices((ny, nx))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    r_max = r.max()
    radial_mean = np.array([psd2d[r == i].mean() if np.any(r == i) else 0 for i in range(r_max + 1)])
    return radial_mean


def spectral_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    RAPSD egrileri arasindaki farkin karesi, zaman boyunca ortalanmis.
    original/reconstructed sekli: (n_lat, n_lon, n_time) -- bizim genel konvansiyonumuz.
    """
    n_t = original.shape[-1]
    errors = []
    for t in range(n_t):
        s_orig = radially_averaged_psd(original[:, :, t])
        s_recon = radially_averaged_psd(reconstructed[:, :, t])
        min_len = min(len(s_orig), len(s_recon))
        errors.append(np.sum((s_orig[:min_len] - s_recon[:min_len]) ** 2))
    return float(np.mean(errors))


def evaluate_against_bounds(rmse: float, bounds: dict) -> dict:
    """RMSE'nin her esigi gecip gecmedigini (True/False) ve esigin kaç katı oldugunu dondurur."""
    result = {}
    for level in ["b_low", "b_mid", "b_high"]:
        bound = bounds[level]
        result[f"passes_{level}"] = bool(rmse <= bound)
        result[f"{level}_ratio"] = float(rmse / bound) if bound > 0 else float("nan")
    return result
