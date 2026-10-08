"""
Ortak yöntem arayüzü.

Neden bu var: run_experiment.py, PCA/SIREN/hash-grid arasındaki farkı
bilmek ZORUNDA kalmasın diye. Her yöntem bu iki metodu sağladığı sürece
(fit ve reconstruct), orchestration kodu değişmeden kalır.

Bu, C#'taki interface / abstract class mantığının Python karşılığı:
ABC (Abstract Base Class) kullanıyoruz ki bir yöntem bu metodları
implemente etmeden kullanılmaya çalışılırsa Python hemen hata versin,
sessizce yanlış davranmasın.
"""

from abc import ABC, abstractmethod
import numpy as np


class CompressionMethod(ABC):
    """Her yöntem (PCA, SIREN, hash-grid) bu sınıftan türeyecek."""

    def __init__(self, **hyperparams):
        """
        hyperparams: config dosyasından gelen yöntem-özel parametreler
        (örn. SIREN için w0/depth, PCA için n_components).
        Method-agnostic kalması için burada spesifik bir şey varsaymıyoruz.
        """
        self.hyperparams = hyperparams

    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """
        'Encode' adımı. Veriyi sıkıştırılmış/öğrenilmiş temsile çevirir.
        Örn. SIREN için: ağı eğit. PCA için: bileşenleri hesapla.
        Zaman ölçümü BU fonksiyonun çağrısını sarmalayacak.
        """
        raise NotImplementedError

    @abstractmethod
    def reconstruct(self) -> np.ndarray:
        """
        'Decode' adımı. Sıkıştırılmış temsilden orijinal veriyi geri üretir.
        Zaman ölçümü BU fonksiyonun çağrısını sarmalayacak.
        """
        raise NotImplementedError

    @abstractmethod
    def compression_ratio(self, original_data: np.ndarray) -> float:
        """
        Orijinal boyut / sıkıştırılmış boyut oranı.
        Her yöntemde 'sıkıştırılmış boyut'un anlamı farklı olduğu için
        (PCA'da bileşen sayısı, SIREN'de ağ ağırlıkları) bunu her
        alt sınıf kendi mantığıyla hesaplar.
        """
        raise NotImplementedError
