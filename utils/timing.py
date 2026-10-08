"""
Ortak ölçüm harness'i: encode/decode süresini ve GPU bellek kullanımını
her yöntem için AYNI şekilde ölçmemizi sağlar.

Neden context manager (`with` bloğu)?
Çünkü "şu kod bloğunun süresini ve peak memory'sini ölç" işlemini
tek satırda, tekrar tekrar yazmadan yapabiliyoruz. Örnek kullanım
aşağıda `if __name__ == "__main__"` bloğunda var.
"""

import time
import torch


class Measurement:
    """Tek bir ölçümün sonucunu tutan basit bir veri sınıfı."""

    def __init__(self):
        self.elapsed_seconds = None
        self.peak_memory_mb = None


class measure_block:
    """
    Context manager: bir kod bloğunun wall-clock süresini ve
    (GPU varsa) peak memory kullanımını ölçer.

    Kullanım:
        result = Measurement()
        with measure_block(result, device="cuda"):
            model.fit(data)
        print(result.elapsed_seconds, result.peak_memory_mb)
    """

    def __init__(self, result: Measurement, device: str = "cpu"):
        self.result = result
        self.device = device

    def __enter__(self):
        # Bellek sayaçlarını sıfırla ki önceki koşulardan kalan
        # bir şey ölçümü bozmasın
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()  # GPU'nun bekleyen işlemlerini bitir

        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()  # süreyi ölçmeden önce GPU'yu bekle
            self.result.peak_memory_mb = (
                torch.cuda.max_memory_allocated() / (1024 ** 2)
            )
        else:
            self.result.peak_memory_mb = None

        self.result.elapsed_seconds = time.perf_counter() - self._start
        return False  # exception varsa yutma, yükselt


if __name__ == "__main__":
    # Küçük bir sanity-check: harness gerçekten çalışıyor mu?
    result = Measurement()
    with measure_block(result, device="cpu"):
        time.sleep(0.5)
    print(f"Ölçülen süre: {result.elapsed_seconds:.3f} sn (beklenen ~0.5 sn)")
