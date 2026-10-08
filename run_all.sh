#!/bin/bash
# Tum deney matrisini (3 domain x 3 yontem) sirayla calistirir.
# Mudahale gerektirmez -- bir config hata verirse loglar, bir sonrakine gecer.
#
# Kullanim (tmux icinde):
#   tmux new -s deneyler
#   cd experiments
#   bash run_all.sh
#   [Ctrl+B, sonra D ile ayril]
#
# Daha sonra durumu kontrol etmek icin:
#   tmux attach -t deneyler
#   veya: tail -f results/logs/_overall.log

set -uo pipefail   # tanimsiz degiskenlerde ve pipe hatalarinda uyar, ama
                    # 'set -e' KULLANMIYORUZ bilerek -- bir config patlarsa
                    # script tamamen durmasin, sadece o config atlansin.

mkdir -p results/logs

OVERALL_LOG="results/logs/_overall.log"
echo "=== Tum deney matrisi basliyor: $(date) ===" | tee "$OVERALL_LOG"

# Siralama bilinclidir: kucuk->buyuk domain, her domain icinde de
# en hizlidan en yavasa (pca -> siren -> hashgrid). Boylece erken bir
# hata olsa bile elinde en azindan hizli/ucuz kosularin sonucu olur.
CONFIGS=(
  "configs/small_pca.yaml"
  "configs/small_siren.yaml"
  "configs/small_hashgrid.yaml"
  "configs/medium_pca.yaml"
  "configs/medium_siren.yaml"
  "configs/medium_hashgrid.yaml"
  "configs/large_pca.yaml"
  "configs/large_siren.yaml"
  "configs/large_hashgrid.yaml"
)

SUCCESS_COUNT=0
FAIL_COUNT=0

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  log_file="results/logs/${name}.log"

  echo "" | tee -a "$OVERALL_LOG"
  echo "=== [$name] basladi: $(date) ===" | tee -a "$OVERALL_LOG"

  # Her config'in kendi log dosyasi var. 2>&1 hata mesajlarini da
  # ayni akisa dahil ediyor, tee hem ekrana hem dosyaya yaziyor.
  python run_experiment.py --config "$cfg" 2>&1 | tee "$log_file"

  # PIPESTATUS[0]: pipe'in ilk komutunun (python'un) gercek exit kodu.
  # tee'nin kendi exit kodu her zaman 0 donuyor, o yuzden dogrudan $? kullanamayiz.
  exit_code=${PIPESTATUS[0]}

  if [ "$exit_code" -eq 0 ]; then
    echo "=== [$name] BASARILI: $(date) ===" | tee -a "$OVERALL_LOG"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
  else
    echo "=== [$name] HATA (exit code $exit_code): $(date) -- devam ediliyor ===" | tee -a "$OVERALL_LOG"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
done

echo "" | tee -a "$OVERALL_LOG"
echo "=== Tum deneyler bitti: $(date) ===" | tee -a "$OVERALL_LOG"
echo "Basarili: $SUCCESS_COUNT / ${#CONFIGS[@]}  |  Hatali: $FAIL_COUNT" | tee -a "$OVERALL_LOG"

# En az bir koşu başarılıysa raporu otomatik üret -- sen geri döndüğünde
# hem ham sonuçlar hem hazır rapor seni bekliyor olsun.
if [ "$SUCCESS_COUNT" -gt 0 ]; then
  echo "" | tee -a "$OVERALL_LOG"
  echo "=== Rapor uretiliyor: $(date) ===" | tee -a "$OVERALL_LOG"
  python -m utils.report 2>&1 | tee -a "$OVERALL_LOG"
else
  echo "Hicbir kosu basarili olmadigi icin rapor uretilmedi." | tee -a "$OVERALL_LOG"
fi

echo "" | tee -a "$OVERALL_LOG"
echo "=== TAMAMLANDI: $(date) ===" | tee -a "$OVERALL_LOG"
