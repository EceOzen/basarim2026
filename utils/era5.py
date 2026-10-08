"""
ERA5 veri yükleme -- earth_systems_playground_part4 notebook'undaki
Cell 6'dan (build_grid, fetch_era5_t2m, to_lat_lon_time) birebir taşındı.

Notebook'takiyle FARKI: burada Colab Secrets değil, ortam değişkenleri
kullanıyoruz (CDS_API_URL / CDS_API_KEY) -- çünkü bu kod artık Colab
dışında (GCP VM gibi) da çalışacak. VM'de çalıştırmadan önce:
    export CDS_API_KEY="senin-api-anahtarın"
şeklinde ayarlaman yeterli (ya da ~/.cdsapirc dosyan zaten varsa
hiçbir şey yapmana gerek yok, cdsapi.Client() otomatik onu kullanır).
"""

import os
import numpy as np


def build_grid(region: dict, res: float):
    """Notebook Cell 6 ile birebir aynı."""
    lats = np.arange(region["lat_min"], region["lat_max"] + res, res)
    lons = np.arange(region["lon_min"], region["lon_max"] + res, res)
    return lats, lons


def to_lat_lon_time(arr, data_lats=None, data_lons=None):
    """Notebook Cell 6 ile birebir aynı: (time, lat, lon) sırasını
    (lat, lon, time)'a çevirir, gerekiyorsa."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and data_lats is not None and data_lons is not None:
        if arr.shape[1] == len(data_lats) and arr.shape[2] == len(data_lons):
            arr = arr.transpose(1, 2, 0)
    return arr


def _cache_path(cache_dir: str, prefix: str, region: dict, date_range: tuple, res: float) -> str:
    tag = (
        f"{prefix}_{region['lat_min']}_{region['lat_max']}_{region['lon_min']}_"
        f"{region['lon_max']}_{date_range[0]}_{date_range[1]}_{res}"
    ).replace(".", "p")
    return os.path.join(cache_dir, tag + ".nc")


def fetch_era5_t2m(region: dict, date_range: tuple, res: float,
                    cache_dir: str = "era5_cache", use_cache: bool = True):
    """Notebook Cell 6 (fetch_era5_t2m) ile birebir aynı mantık, sadece
    Colab Secrets yerine ortam değişkeni / ~/.cdsapirc kullanıyor.
    Returns (values, lats, lons) -- values şekli (lat, lon, time)."""
    import xarray as xr

    os.makedirs(cache_dir, exist_ok=True)
    nc_path = _cache_path(cache_dir, "t2m", region, date_range, res)

    if not (use_cache and os.path.exists(nc_path)):
        import cdsapi

        # CDS_API_URL genelde hep aynı (tek bir resmi endpoint var), o yüzden
        # burada sabit bir varsayılan tanımlıyoruz -- kullanıcı sadece
        # CDS_API_KEY ayarlasa bile çalışsın diye. CDS_API_URL ortam
        # değişkeniyle bilerek override edilebilir (nadiren gerekir).
        cds_url = os.environ.get("CDS_API_URL", "https://cds.climate.copernicus.eu/api")
        cds_key = os.environ.get("CDS_API_KEY")
        c = cdsapi.Client(url=cds_url, key=cds_key) if cds_key else cdsapi.Client()
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": "2m_temperature",
                "date": f"{date_range[0]}/{date_range[1]}",
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": [region["lat_max"], region["lon_min"], region["lat_min"], region["lon_max"]],
                "grid": [res, res],
                "format": "netcdf",
            },
            nc_path,
        )
        print(f"[era5] indirildi ve önbelleklendi: {nc_path}")
    else:
        print(f"[era5] önbellekten kullanılıyor: {nc_path}")

    ds = xr.open_dataset(nc_path)
    da = ds["t2m"] - 273.15  # Kelvin -> Celsius

    time_dim = "time" if "time" in da.dims else "valid_time"
    da = da.transpose("latitude", "longitude", time_dim)

    return da.values, da["latitude"].values, da["longitude"].values


def load_era5_field(region: dict, date_range: tuple, res: float, cache_dir: str = "era5_cache") -> np.ndarray:
    """Tek satırda kullanılabilir yardımcı: bölge+tarih+çözünürlük al,
    (n_lat, n_lon, n_time) şeklinde bir numpy array döndür."""
    raw, data_lats, data_lons = fetch_era5_t2m(region, date_range, res, cache_dir=cache_dir)
    return to_lat_lon_time(raw, data_lats, data_lons)


def fetch_era5_ensemble_spread(region: dict, date_range: tuple, res: float,
                                cache_dir: str = "era5_cache", use_cache: bool = True) -> np.ndarray:
    """
    ClimateBenchPress error-bound hesaplaması için: ERA5'in ensemble spread
    (EDA - Ensemble of Data Assimilations) ürünü, product_type='ensemble_spread'
    ile isteniyor. Bu, ERA5'in KENDİ belirsizlik tahmini -- error bound'u
    buradan türetiyoruz (bkz. utils/climatebench.py).

    NOT: EDA ürünü orijinalde daha düşük çözünürlükte (3 saatlik, ~63km) üretiliyor,
    CDS bunu talep edilen grid'e interpolate ediyor -- notebook'takiyle aynı davranış.
    """
    os.makedirs(cache_dir, exist_ok=True)
    nc_path = _cache_path(cache_dir, "t2m_spread", region, date_range, res)

    if not (use_cache and os.path.exists(nc_path)):
        import cdsapi

        cds_url = os.environ.get("CDS_API_URL", "https://cds.climate.copernicus.eu/api")
        cds_key = os.environ.get("CDS_API_KEY")
        c = cdsapi.Client(url=cds_url, key=cds_key) if cds_key else cdsapi.Client()
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "ensemble_spread",
                "variable": "2m_temperature",
                "date": f"{date_range[0]}/{date_range[1]}",
                "time": [f"{h:02d}:00" for h in range(0, 24, 3)],  # EDA 3 saatlik
                "area": [region["lat_max"], region["lon_min"], region["lat_min"], region["lon_max"]],
                "grid": [res, res],
                "format": "netcdf",
            },
            nc_path,
        )
        print(f"[era5] ensemble spread indirildi ve önbelleklendi: {nc_path}")
    else:
        print(f"[era5] ensemble spread önbellekten kullanılıyor: {nc_path}")

    import xarray as xr
    ds = xr.open_dataset(nc_path)
    da = ds["t2m"]  # spread zaten Kelvin farkı olarak geliyor, -273.15 YAPILMAZ

    time_dim = "time" if "time" in da.dims else "valid_time"
    da = da.transpose("latitude", "longitude", time_dim)
    return to_lat_lon_time(da.values, da["latitude"].values, da["longitude"].values)
