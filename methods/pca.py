"""
PCA yöntemi — earth_systems_playground_part4 notebook'undaki mantığın
(Cell 43: X_B = field_B.reshape(n_lat*n_lon, n_time).T) sınıf haline
getirilmiş hali.

ÖNEMLİ TASARIM NOKTASI (notebook'tan doğrudan geliyor):
Veri (n_lat, n_lon, n_time) şeklinde geliyor ama PCA "örnek x özellik"
matrisi bekler. Notebook'taki seçim: HER ZAMAN ADIMINI BİR ÖRNEK,
her pikseli bir ÖZELLİK olarak almak (transpose edip flatten ederek).
Yani PCA, zaman boyunca tekrar eden mekansal örüntüleri (spatial modes)
öğreniyor. Bu satırı değiştirirsen (örn. mekan örnek, zaman özellik
yaparsan) sonuçlar da tamamen değişir -- bilinçli bir karar, tesadüf değil.

hyperparams'tan beklenen: n_components (kaç bileşen kullanılacak)
"""

import numpy as np
from sklearn.decomposition import PCA as SklearnPCA

from .base import CompressionMethod


class PCAMethod(CompressionMethod):
    def fit(self, data: np.ndarray) -> None:
        # data şekli: (n_lat, n_lon, n_time) -- notebook'taki field_B ile aynı konvansiyon
        n_lat, n_lon, n_time = data.shape
        self._original_shape = data.shape

        # Notebook Cell 43 ile birebir aynı: (n_lat*n_lon, n_time) -> transpose -> (n_time, n_lat*n_lon)
        # Yani satırlar = zaman adımları, sütunlar = piksel değerleri
        self._X = data.reshape(n_lat * n_lon, n_time).T

        n_components = self.hyperparams["n_components"]
        # n_components, örnek sayısından (n_time) fazla olamaz -- notebook'ta bu
        # component_counts listesini süzerken zaten kontrol ediliyordu (Cell 43),
        # burada da aynı güvenliği koruyoruz.
        n_components = min(n_components, self._X.shape[0], self._X.shape[1])

        self.pca = SklearnPCA(n_components=n_components)
        self._X_reduced = self.pca.fit_transform(self._X)

    def reconstruct(self) -> np.ndarray:
        # inverse_transform, (n_time, n_lat*n_lon) formunda geri veriyor.
        # Bunu tekrar (n_lat, n_lon, n_time)'a çevirmemiz lazım -- fit'teki
        # dönüşümün TAM TERSİ (reshape sırası önemli, notebook'ta da bu sıra kullanıldı).
        X_reconstructed = self.pca.inverse_transform(self._X_reduced)
        n_lat, n_lon, n_time = self._original_shape
        return X_reconstructed.T.reshape(n_lat, n_lon, n_time)

    def compression_ratio(self, original_data: np.ndarray) -> float:
        # Notebook Cell 43 ile birebir aynı hesap:
        # sıkıştırılmış temsil = bileşenler (components_) + ortalama vektör (mean_)
        # + düşük boyutlu izdüşümler (X_reduced). Hepsi float32 (4 byte) varsayılıyor.
        original_size_bytes = original_data.size * 4
        pca_size_bytes = (
            self.pca.components_.size + self.pca.mean_.size + self._X_reduced.size
        ) * 4
        return original_size_bytes / pca_size_bytes
