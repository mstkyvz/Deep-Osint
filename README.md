# Deep-Osint
Finding the location of any photo with Deep Learning

# Currently Only Active For Turkey

I imported 783,562 km² of imagery to Google Earth Engine (GEE)

No matter what you put in any forest, mountain, stone, street photo in Turkey, it finds the location in maximum 30 minutes.

There are 1,182,964 images in the train set.

contact me to contribute to the project : mustafawalker@protonmail.com

---

## Sokak seviyesi konum bulma hattı (`deeposint`)

Şehir içinde çekilmiş bir fotoğrafın **hangi caddede, hangi noktada ve hangi yöne bakılarak** çekildiğini bulur. Büyük bir LLM kullanmaz. Kararları script'ler ve yerel çalışan küçük görüntü modelleri verir.

> **Durum:** Kod yazıldı ama **henüz uçtan uca çalıştırılmadı**. Yöntem daha önce elle, adım adım denendi: Karamürsel'de 16.926 Street View görüntüsü arasından doğru nokta ±30 m ile bulundu. Paketlenmiş hâli ise test edilmedi. Yapılacaklar aşağıda.

### Nasıl çalışır

```
foto + bölge (ilçe adı ya da bbox)
   │
   ├─ 0. EXIF'te GPS var mı? ───────────────────────────────► varsa bitti
   │
   ├─ 1. Yollar      OSM Overpass: trunk/primary/secondary/tertiary (+ istersen residential)
   ├─ 2. Örnekleme   her yol boyunca 40 m'de bir nokta + yolun yönü
   ├─ 3. Galeri      Street View: z17 kapsama karolarından panoları bul,
   │                 her panoda yola dik iki yöne ±35° açılarla görüntü al
   ├─ 4. Recall      MegaLoc (ya da DINOv2) global vektörleri → en benzer 200 görüntü
   ├─ 5. Doğrulama   SuperPoint + LightGlue (ya da SIFT) → RANSAC ile tutarlı eşleşme sayısı
   │                 (alttaki yol ve bordür bölgesi maskelenir; tekrar eden desenler yanıltmasın)
   ├─ 6. Oylama      adaylar 40 m'lik kümelere toplanır; doğru yer genelde birden çok
   │                 komşu panoda görünür, yanlış eşleşme ise tek bir görüntüde kalır
   └─ 7. Sıralama    sezgisel skor ya da LightGBM (lambdarank) → koordinat + yön + güven
```

| Katman | Varsayılan | Alternatif | Not |
|---|---|---|---|
| Global arama | MegaLoc | DINOv2 ViT-B/14 | `torch.hub` ile ilk kullanımda iner |
| Yerel doğrulama | SuperPoint + LightGlue | OpenCV SIFT | RANSAC (MAGSAC) temel matris |
| Karar | Sezgisel skor | LightGBM lambdarank | Model, galerinin kendisinden etiketsiz eğitilir |
| İpucu çıkarma (isteğe bağlı) | — | Ollama + `qwen2.5vl:7b` | Sadece sabit bir JSON şemasını doldurur, karar vermez |

### Kurulum

```bash
git clone https://github.com/mstkyvz/Deep-Osint && cd Deep-Osint
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"          # torch, LightGlue, LightGBM, HEIC
# sadece hafif çekirdek için: pip install -e .
```

Disk ve bellek ihtiyacı:
- **Modeller:** MegaLoc ve DINOv2 yaklaşık 0,5–1 GB, SuperPoint + LightGlue yaklaşık 50 MB.
- **Galeri:** küçük bir ilçe için 15–20 bin görüntü, yaklaşık 1–1,5 GB.
- **İsteğe bağlı görsel-dil modeli:** `ollama pull qwen2.5vl:7b`, yaklaşık 6 GB.

### Kullanım

```bash
# Tek komut: galeri + arama + doğrulama + sıralama
deeposint -v locate foto.jpg --area "Karamürsel, Kocaeli"

# bbox ile ve hafif ayarlarla (GPU yoksa)
deeposint locate foto.jpg --bbox 40.655,29.53,40.725,29.70 --retrieval dinov2 --verify sift

# Sorgunun sadece bir bölgesini kullan (ör. yol yerine binalar)
deeposint locate foto.jpg --area "Karamürsel, Kocaeli" --box 0,450,1452,1100

# Sadece galeriyi önceden hazırla (internet tarafı)
deeposint gallery --area "Karamürsel, Kocaeli" --work runs/karamursel

# Yerel görsel-dil modeliyle ipucu çıkar (Ollama çalışıyor olmalı)
deeposint clues foto.jpg --model qwen2.5vl:7b

# Sıralayıcıyı galeriden etiketsiz eğit ve kullan
deeposint train-reranker --work runs/karamursel --out reranker.txt
deeposint locate foto.jpg --area "Karamürsel, Kocaeli" --reranker reranker.txt
```

Çıktılar `runs/<bölge>/` klasörüne yazılır:
- `result.json`: koordinat, yön, cadde, güven seviyesi
- `ranking.json`: ilk 20 küme ve her birinin özellikleri
- `top.jpg`: sorgu fotoğrafı ve en iyi adayların yan yana görüntüsü
- `gallery/`: indirilen Street View görüntüleri ve `views.json`

`result.json` biçimi (değerler temsilîdir, gerçek bir çalıştırmanın çıktısı değildir):

```json
{
  "method": "streetview",
  "lat": 40.6905, "lon": 29.6083, "heading": 8.0,
  "road": "Kocaeli Yolu",
  "confidence": {"level": "high", "margin": 1.7, "top_inliers": 64, "support_panos": 4}
}
```

### Proje yapısı

```
deeposint/
  cli.py         komut satırı (locate, gallery, clues, train-reranker)
  pipeline.py    uçtan uca akış
  osm.py         Nominatim (bölge → bbox), Overpass (yollar), yol örnekleme
  streetview.py  kapsama karoları → panolar → yola dik görüntüler
  retrieval.py   MegaLoc / DINOv2 vektörleri, en benzer K
  verify.py      SuperPoint+LightGlue / SIFT + RANSAC
  rerank.py      kümeleme, oylama, sezgisel/LightGBM skor, etiketsiz eğitim verisi
  clues.py       Ollama görsel-dil modeli ile şemaya bağlı ipucu çıkarma
  imageio.py     görüntü yükleme (HEIC dahil), EXIF GPS
  report.py      sonuç görseli
  geo.py, http.py  yardımcılar (mesafe/yön, önbellekli HTTP)
```

### Yapılacaklar

**Çalıştırma ve doğrulama (öncelikli)**
- [ ] Temiz bir ortamda `pip install -e ".[all]"` ve `deeposint locate` uçtan uca çalıştırılacak.
- [ ] MegaLoc `torch.hub` giriş noktası (`get_trained_model`) ve çıktı biçimi doğrulanacak (`retrieval.py`'de TODO).
- [ ] LightGlue'nun MPS (Apple Silicon) üzerinde çalıştığı ve hızı kontrol edilecek.
- [ ] Karamürsel vakası regresyon testi olacak: doğru nokta ilk sırada çıkmalı, güven "high" olmalı.
- [ ] Oylama ve skor eşikleri (`min_inliers`, güven sınırları) gerçek sorgularla ayarlanacak.

**Doğruluk**
- [ ] Sorgu için sadece sabit yapıların (bina, çatı, ağaç gövdesi) kullanıldığı otomatik maske (gökyüzü, yol, araç segmentasyonu).
- [ ] Street View'un eski tarihli çekimlerinin de galeriye eklenmesi (mevsim farkı).
- [ ] Güneş ve gölge yönünden kamera yönü aralığı çıkarılıp ters yöndeki görüntülerin elenmesi.
- [ ] Sıralayıcı eğitim verisine gerçek telefon fotoğraflarının eklenmesi (şu an sorgu da Street View, alan farkını küçümsüyor).
- [ ] Kesin konum için en iyi panolardan PnP / üçgenleme ile metre seviyesinde kamera konumu.

**Kapsam ve ölçek**
- [ ] Bölge bilinmiyorsa GeoCLIP / StreetCLIP ile kaba konum tahmini ve arama alanı önerisi.
- [ ] Büyük alanlar için FAISS indeksi ve parça parça galeri.
- [ ] Doğa ve dağ fotoğrafları için DEM tabanlı silüet eşleme modülü (SRTM / AWS Terrain Tiles).
- [ ] Yazı ve tabela okuma (OCR) → OSM'de isim araması → aday nokta.

**Sağlayıcılar**
- [ ] Street View erişimi şu an Google Maps web istemcisinin belgelenmemiş uç noktalarını kullanıyor; bunlar değişebilir ya da engellenebilir. Seçilebilir sağlayıcı katmanı: Street View Static API (anahtarlı) ve Mapillary.
- [ ] İstek hızı sınırlama ve yeniden deneme politikası.

**Altyapı**
- [ ] Birim testler (örnekleme, kapsama karosu ayrıştırma, kümeleme) + CI.
- [ ] Dockerfile.

### Bilinen sınırlar

- **Arama alanı daralmadan çalışmaz.** İlçe ya da semt düzeyinde bir bölge gerekir. Bütün bir il saatler sürer.
- **Tip projeler** (aynı TOKİ bloğu onlarca ilde var) yanlış eşleşmeye açık. Komşu pano oylaması bunu azaltır ama sıfırlamaz.
- **Street View çekimi eskiyse** (yeni bina, yeni korkuluk) puanlar düşer.
- Kırsal, doğa ve iç mekân fotoğrafları bu hattın kapsamında değil.

### Sorumlu kullanım

Kendi fotoğraflarınızda ya da analiz izniniz olan görüntülerde kullanın. İnsanların rızası olmadan nerede yaşadığını veya bulunduğunu tespit etmek için kullanmayın. Google, OpenStreetMap (ODbL) ve Nominatim kullanım koşullarına uyun; Nominatim için saniyede en fazla 1 istek gönderin.
