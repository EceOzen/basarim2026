"""
Hash-grid yöntemi (Instant-NGP tarzı, Müller et al. 2022, multiresolution
hash encoding) -- saf PyTorch implementasyonu, tiny-cuda-nn gerektirmez.

MANTIK (SIREN'den farkı):
SIREN, "koordinat -> değer" fonksiyonunu tamamen ağ ağırlıklarında
öğrenir. Hash-grid ise temsil kapasitesinin çoğunu ÖĞRENİLEBİLİR BİR
ARAMA TABLOSUNDA (hash table) tutar:
  1. Koordinat uzayı L farklı çözünürlükte (kaba->ince) grid'lere bölünür
  2. Her seviyede, koordinatın düştüğü hücrenin köşeleri bir hash
     fonksiyonuyla sabit boyutlu bir tabloya eşlenir (çakışma/collision
     olabilir, bilerek -- tablo küçük tutuluyor, bu "sıkıştırma" kısmı)
  3. Köşe feature'ları (F boyutlu) interpolate edilir
  4. L seviyenin feature'ları birleştirilip küçük bir MLP'ye verilir

hyperparams'tan beklenen:
  n_levels (L, örn. 8), n_features_per_level (F, örn. 2),
  log2_hashmap_size (örn. 14 -> 2^14 = 16384 satırlık tablo/seviye),
  base_resolution (en kaba grid, örn. 4), finest_resolution (en ince grid, örn. 128),
  mlp_hidden_dim, n_epochs, seed, lr (opsiyonel), batch_size (opsiyonel)
"""

import numpy as np
import torch
import torch.nn as nn

from .base import CompressionMethod


class MultiresHashEncoding(nn.Module):
    """
    L seviyeli, her seviyesi kendi hash tablosuna sahip encoding katmanı.
    3 boyutlu koordinat (lat, lon, time) alır, L*F boyutlu feature döndürür.
    """

    def __init__(self, n_levels, n_features_per_level, log2_hashmap_size,
                 base_resolution, finest_resolution, in_dim=3):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.hashmap_size = 2 ** log2_hashmap_size
        self.in_dim = in_dim

        # Instant-NGP Eq. (3): seviyeler arası çözünürlük geometrik olarak artar.
        # b: ardışık seviyeler arası büyüme oranı.
        if n_levels > 1:
            b = np.exp((np.log(finest_resolution) - np.log(base_resolution)) / (n_levels - 1))
        else:
            b = 1.0
        self.resolutions = [int(np.floor(base_resolution * (b ** l))) for l in range(n_levels)]

        # Her seviye için ayrı, öğrenilebilir bir hash tablosu (embedding).
        # Küçük başlatma (Instant-NGP'de olduğu gibi) -- büyük başlangıç
        # değerleri erken eğitimde patlamaya yol açabiliyor.
        self.tables = nn.ModuleList([
            nn.Embedding(self.hashmap_size, n_features_per_level)
            for _ in range(n_levels)
        ])
        for table in self.tables:
            nn.init.uniform_(table.weight, -1e-4, 1e-4)

        # Sabit büyük asal sayılar (Instant-NGP Eq. 4) -- boyutlar arası
        # hash çakışmalarını azaltmak için kullanılıyor, öğrenilmiyor.
        primes = torch.tensor([1, 2654435761, 805459861], dtype=torch.int64)
        self.register_buffer("primes", primes[:in_dim])

    def _hash(self, grid_coords: torch.Tensor) -> torch.Tensor:
        """grid_coords: (N, in_dim) tam sayı grid indeksleri -> (N,) hash indeksi."""
        h = torch.zeros(grid_coords.shape[0], dtype=torch.int64, device=grid_coords.device)
        for d in range(self.in_dim):
            h = h ^ (grid_coords[:, d].to(torch.int64) * self.primes[d])
        return h % self.hashmap_size

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (N, in_dim), [-1, 1] aralığında normalize edilmiş. Döndürür: (N, n_levels * n_features_per_level)."""
        coords01 = (coords + 1) / 2  # [-1,1] -> [0,1]
        outputs = []

        for level, resolution in enumerate(self.resolutions):
            scaled = coords01 * (resolution - 1)
            floor = torch.floor(scaled)
            frac = scaled - floor  # trilinear interpolasyon ağırlıkları

            # 3D'de bir hücrenin 8 köşesi var (2^in_dim). Her köşe için
            # hash'le -> feature al -> interpolasyon ağırlığıyla topla.
            n_corners = 2 ** self.in_dim
            level_feature = torch.zeros(
                coords.shape[0], self.n_features_per_level, device=coords.device
            )
            for corner in range(n_corners):
                offset = torch.tensor(
                    [(corner >> d) & 1 for d in range(self.in_dim)],
                    device=coords.device, dtype=torch.float32,
                )
                corner_coords = torch.clamp(floor + offset, min=0, max=resolution - 1)
                weight = torch.prod(
                    torch.where(offset == 1, frac, 1 - frac), dim=-1, keepdim=True
                )
                idx = self._hash(corner_coords)
                level_feature = level_feature + weight * self.tables[level](idx)

            outputs.append(level_feature)

        return torch.cat(outputs, dim=-1)


class HashGridMLP(nn.Module):
    """Encoding + küçük MLP head -- Instant-NGP'nin genel tasarımı."""

    def __init__(self, encoding: MultiresHashEncoding, hidden_dim=64, out_dim=1):
        super().__init__()
        self.encoding = encoding
        in_dim = encoding.n_levels * encoding.n_features_per_level
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, coords):
        return self.mlp(self.encoding(coords))


def _count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


class HashGridMethod(CompressionMethod):
    def fit(self, data: np.ndarray) -> None:
        n_lat, n_lon, n_time = data.shape
        self._original_shape = data.shape

        hp = self.hyperparams
        seed = hp.get("seed", 0)
        n_levels = hp.get("n_levels", 8)
        n_features_per_level = hp.get("n_features_per_level", 2)
        log2_hashmap_size = hp.get("log2_hashmap_size", 14)
        base_resolution = hp.get("base_resolution", 4)
        finest_resolution = hp.get("finest_resolution", 128)
        mlp_hidden_dim = hp.get("mlp_hidden_dim", 64)
        n_epochs = hp.get("n_epochs", 20000)
        batch_size = hp.get("batch_size", 16384)
        lr = hp.get("lr", 1e-2)  # hash-grid'ler genelde SIREN'den daha yüksek lr ile hızlı yakınsıyor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Koordinat/değer hazırlığı: siren.py ile birebir aynı normalizasyon
        # (tutarlılık için -- iki yöntemi karşılaştırırken aynı ön-işleme kullanılmalı)
        lat_idx = np.arange(n_lat)
        lon_idx = np.arange(n_lon)
        time_idx = np.arange(n_time)
        LAT, LON, TIME = np.meshgrid(lat_idx, lon_idx, time_idx, indexing="ij")
        lat_norm = 2 * (LAT - lat_idx.min()) / max(lat_idx.max() - lat_idx.min(), 1) - 1
        lon_norm = 2 * (LON - lon_idx.min()) / max(lon_idx.max() - lon_idx.min(), 1) - 1
        time_norm = 2 * (TIME - time_idx.min()) / max(time_idx.max() - time_idx.min(), 1) - 1

        coords = np.stack([lat_norm.ravel(), lon_norm.ravel(), time_norm.ravel()], axis=-1)
        values = data.ravel()

        self._val_mean = float(values.mean())
        self._val_std = float(values.std()) or 1.0
        values_norm = (values - self._val_mean) / self._val_std

        self._coords_t = torch.tensor(coords, dtype=torch.float32, device=self.device)
        values_t = torch.tensor(values_norm, dtype=torch.float32, device=self.device).unsqueeze(-1)

        torch.manual_seed(seed)
        encoding = MultiresHashEncoding(
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
        )
        self.model = HashGridMLP(encoding, hidden_dim=mlp_hidden_dim).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        n_pts = self._coords_t.shape[0]
        bs = min(batch_size, n_pts)

        for epoch in range(n_epochs):
            idx = torch.randint(0, n_pts, (bs,), device=self.device)
            optimizer.zero_grad()
            pred = self.model(self._coords_t[idx])
            loss = nn.functional.mse_loss(pred, values_t[idx])
            loss.backward()
            optimizer.step()
            scheduler.step()

    def reconstruct(self) -> np.ndarray:
        chunk = 50000
        n_pts = self._coords_t.shape[0]
        preds = []
        with torch.no_grad():
            for i in range(0, n_pts, chunk):
                preds.append(self.model(self._coords_t[i:i + chunk]).cpu().numpy())
        recon_norm = np.concatenate(preds, axis=0).ravel()
        recon = recon_norm * self._val_std + self._val_mean
        return recon.reshape(self._original_shape)

    def compression_ratio(self, original_data: np.ndarray) -> float:
        # Sıkıştırılmış temsil = hash tabloları (her tablo hashmap_size x n_features_per_level)
        # + MLP parametreleri. İkisi de öğrenilebilir, ikisi de "sıkıştırılmış boyut"a dahil.
        original_size_bytes = original_data.size * 4
        model_size_bytes = _count_params(self.model) * 4
        return original_size_bytes / model_size_bytes
