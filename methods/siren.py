"""
SIREN yöntemi — earth_systems_playground_part4_with_partC notebook'undaki
SIREN sınıfı + train_siren + evaluate_siren mantığının sınıf haline
getirilmiş hali (Cell 8, 39, 66).

w0=5 STABİLİTE BULGUSU:
Notebook Section C.5'te (Cell 66-67), w0=30 (varsayılan SIREN init)
hidden_dim=96, n_layers=5 gibi derin ağlarda ciddi eğitim kararsızlığına
yol açıyor. w0'ı 5'e düşürmek bu kararsızlığı çözüyor. Bu yüzden burada
DEFAULT_W0=5 -- config'te başka bir şey yazmazsan güvenli tarafta kalırsın.

hyperparams'tan beklenen: w0 (opsiyonel, default 5), depth (n_layers),
hidden_dim, n_epochs, seed, lr (opsiyonel), batch_size (opsiyonel)
"""

import numpy as np
import torch
import torch.nn as nn

from .base import CompressionMethod

DEFAULT_W0 = 5.0


class SIREN(nn.Module):
    """Notebook Cell 8 ile birebir aynı: sinüzoidal aktivasyonlu MLP (Sitzmann et al., 2020)."""

    def __init__(self, in_dim, hidden_dim=64, n_layers=4, out_dim=1, w0=DEFAULT_W0):
        super().__init__()
        self.w0 = w0
        layers = [nn.Linear(in_dim, hidden_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.hidden_layers = nn.ModuleList(layers)
        self.out_layer = nn.Linear(hidden_dim, out_dim)
        self._siren_init()

    def _siren_init(self):
        # SIREN'e özgü ağırlık başlatma (Sitzmann et al. Sec 3.2) -- w0'a bağlı
        # bound hesaplaması, standart PyTorch init'inden FARKLI, bilerek böyle.
        with torch.no_grad():
            first = True
            for layer in self.hidden_layers:
                fan_in = layer.weight.shape[1]
                bound = (1 / fan_in) if first else (np.sqrt(6 / fan_in) / self.w0)
                layer.weight.uniform_(-bound, bound)
                first = False
            fan_in = self.out_layer.weight.shape[1]
            self.out_layer.weight.uniform_(
                -np.sqrt(6 / fan_in) / self.w0, np.sqrt(6 / fan_in) / self.w0
            )

    def forward(self, coords):
        x = coords
        for layer in self.hidden_layers:
            x = torch.sin(self.w0 * layer(x))
        return self.out_layer(x)


def _count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


class SIRENMethod(CompressionMethod):
    def fit(self, data: np.ndarray) -> None:
        # data şekli: (n_lat, n_lon, n_time) -- notebook konvansiyonuyla aynı
        n_lat, n_lon, n_time = data.shape
        self._original_shape = data.shape

        hp = self.hyperparams
        seed = hp.get("seed", 0)
        w0 = hp.get("w0", DEFAULT_W0)
        hidden_dim = hp["hidden_dim"]
        n_layers = hp["depth"]
        n_epochs = hp.get("n_epochs", 20000)
        batch_size = hp.get("batch_size", 16384)
        lr = hp.get("lr", 2e-3)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Koordinat + değer hazırlığı: notebook Cell 37 ile birebir aynı mantık ---
        # lat/lon/time indekslerini [-1, 1] aralığına normalize ediyoruz.
        # SIREN'in periyodik (sin) aktivasyonları bu aralıkta en iyi davranıyor;
        # ham (örn. gerçek enlem/boylam) değerlerle eğitmek kararsızlığı artırır.
        lat_idx = np.arange(n_lat)
        lon_idx = np.arange(n_lon)
        time_idx = np.arange(n_time)
        LAT, LON, TIME = np.meshgrid(lat_idx, lon_idx, time_idx, indexing="ij")
        lat_norm = 2 * (LAT - lat_idx.min()) / max(lat_idx.max() - lat_idx.min(), 1) - 1
        lon_norm = 2 * (LON - lon_idx.min()) / max(lon_idx.max() - lon_idx.min(), 1) - 1
        time_norm = 2 * (TIME - time_idx.min()) / max(time_idx.max() - time_idx.min(), 1) - 1

        coords = np.stack([lat_norm.ravel(), lon_norm.ravel(), time_norm.ravel()], axis=-1)
        values = data.ravel()

        # Değer normalizasyonu da notebook ile aynı: (x - mean) / std
        self._val_mean = float(values.mean())
        self._val_std = float(values.std()) or 1.0
        values_norm = (values - self._val_mean) / self._val_std

        self._coords_t = torch.tensor(coords, dtype=torch.float32, device=self.device)
        values_t = torch.tensor(values_norm, dtype=torch.float32, device=self.device).unsqueeze(-1)
        self._values_raw = values  # RMSE'yi orijinal ölçekte hesaplamak için saklıyoruz

        # --- Eğitim: notebook train_siren ile birebir aynı döngü ---
        torch.manual_seed(seed)
        self.model = SIREN(in_dim=3, hidden_dim=hidden_dim, n_layers=n_layers, w0=w0).to(self.device)
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
        # notebook evaluate_siren ile aynı: büyük veri setlerinde belleği patlatmamak
        # için chunk'lar halinde ileri geçiş yapıyoruz.
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
        # Notebook ile aynı: orijinal boyut (float32) / ağ parametre sayısı x 4 byte
        original_size_bytes = original_data.size * 4
        model_size_bytes = _count_params(self.model) * 4
        return original_size_bytes / model_size_bytes
