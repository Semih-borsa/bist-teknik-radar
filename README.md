# BIST Teknik Gözlem Köprüsü

Bu repo, BIST hisselerinde SMA 5/8/13 tabanlı teknik koşulları günlük kapanış verileriyle tarar. Yeni gözlemleri T+1, T+3 ve T+5 işlem günlerinde tekrar değerlendirir. Sonuçları Telegram'a gönderir ve Android uygulamasının okuyacağı `automation/output/feed.json` dosyasını üretir.

Sistem yalnızca teknik gözlem ve arşivleme içindir. Gerçek veya sanal işlem, portföy emri, hedef ya da stop üretmez.

## Takip durumları

- `Teyit aldı`: İlgili stratejinin teknik yapısı sonraki işlem gününde korunmuştur.
- `Kısmi teyit`: Fiyat yapısı korunurken hacim veya ortalama koşullarından biri zayıflamıştır.
- `Teyit gelmedi`: İlk teknik yapı tam olarak korunmamıştır.
- `Tersine döndü`: Fiyat, ilk gözlemin ters yönündeki ortalama bölgesine geçmiştir.
- `Veri bekleniyor`: İlgili T+ günü henüz oluşmamıştır.

Bu etiketler AL/SAT tavsiyesi değil, kurala dayalı teknik durum açıklamasıdır.

## Bir defalık ayarlar

1. Repo, Android uygulamasının anonim feed dosyasını anahtarsız okuyabilmesi için public olmalıdır.
2. `Settings > Secrets and variables > Actions` bölümünde `TELEGRAM_TOKEN` ve `TELEGRAM_CHAT_ID` secret değerleri tanımlanır.
3. `Actions > BIST teknik gözlem > Run workflow` ile ilk tarama başlatılır.
4. Android uygulamasına şu adres kaydedilir:

   `https://raw.githubusercontent.com/Semih-borsa/bist-teknik-radar/main/automation/output/feed.json`

Diğer tarama repoları bu projeden bağımsızdır ve değiştirilmez.
