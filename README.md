# Drone ile Tarim Arazisi Analizi

Bu proje, drone/hava goruntuleri uzerinden tarim alanlarindaki sorun suphelerini
tespit etmek icin hazirlanmis bir makine ogrenmesi projesidir. Sistem iki
asamali calisir:

1. Tarim arazisi siniri semantic segmentation modeli ile tahmin edilir.
2. YOLOv10s modeli yalnizca tarla icinde kalan bolgelerde sorun suphelerini
   isaretler.

Proje kesin zirai teshis vermez. Amac, saha kontrolu gerektirebilecek supheli
bolgeleri on analiz olarak gostermektir.

## Ozellikler

- YOLOv10s ile dort sinifli tarimsal sorun tespiti
- Tarla siniri segmentasyonu ile tarla disi alanlari filtreleme
- Streamlit tabanli web arayuzu
- Guven esigi ayarlama
- Orijinal goruntu ve analiz sonucunu yan yana gosterme
- Sinif bazinda tespit sayisi ve yaklasik alan orani raporu

## Siniflar

| Model sinifi | Turkce yorum | Not |
| --- | --- | --- |
| `planter_skip` | Eksik ekim suphesi | Urun siralarinda bosluk veya cikis eksikligi |
| `double_plant` | Cift veya sik ekim suphesi | Normalden yogun/duzensiz ekim dokusu |
| `weed_cluster` | Duzensiz bitki ortusu | Kesin yabanci ot teshisi degildir |
| `drydown` | Kuruma suphesi | Kesin kuraklik teshisi degildir |

## Proje Yapisi

```text
.
|-- app.py
|-- best.pt
|-- field_boundary_model.pt
|-- requirements.txt
|-- colab_train_full_4class.ipynb
|-- colab_train_field_segmentation.ipynb
|-- demo_images/
|-- demo_images_4class/
|-- scripts/
|   |-- convert_agriculture_vision.py
|   |-- prepare_field_segmentation.py
|   |-- train.py
|   `-- predict.py
`-- tests/
```

## Kurulum

Python 3.10 veya 3.11 onerilir.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Uygulamayi Calistirma

Model agirliklari proje kok dizininde bulunmalidir:

- `best.pt`: dort sinifli YOLOv10s agirligi
- `field_boundary_model.pt`: tarla siniri segmentation agirligi

Streamlit arayuzunu baslatmak icin:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Ardindan tarayicida su adres acilir:

```text
http://localhost:8501
```

## Demo Gorseller

`demo_images/` ve `demo_images_4class/` klasorlerinde uygulamayi hizli denemek
icin ornek goruntuler bulunur. Arayuzde bu goruntulerden biri yuklenerek model
ciktisi test edilebilir.

## Veri Seti

Egitim icin Agriculture-Vision 2017 mini-scale veri seti kullanilmistir.
Veri seti GitHub deposuna eklenmemistir; boyutu buyuktur ve yeniden
indirilebilir.

Resmi kaynaklar:

- https://github.com/SHI-Labs/Agriculture-Vision
- https://registry.opendata.aws/intelinair_agriculture_vision/

Beklenen ham veri yapisi:

```text
data/raw/data2017_miniscale/
|-- field_images/rgb/
|-- field_bounds/
`-- field_labels/
    |-- planter_skip/
    |-- double_plant/
    |-- weed_cluster/
    `-- drydown/
```

Maskeleri YOLO kutu etiketlerine donusturmek icin:

```powershell
python scripts/convert_agriculture_vision.py `
  --source data/raw/data2017_miniscale `
  --output data/processed/agriculture_vision_full_4class
```

Tarla siniri segmentation veri setini hazirlamak icin:

```powershell
python scripts/prepare_field_segmentation.py `
  --source data/raw/data2017_miniscale `
  --output data/processed/field_segmentation
```

## Egitim Notebooklari

- `colab_train_full_4class.ipynb`: YOLOv10s dort sinifli model egitimi
- `colab_train_field_segmentation.ipynb`: tarla siniri segmentation egitimi

## Sonuclar

Son dort sinifli YOLOv10s modelinin dogrulama sonucunda elde edilen genel
degerler:

| Metrik | Deger |
| --- | --- |
| Precision | 0.582 |
| Recall | 0.504 |
| mAP50 | 0.520 |
| mAP50-95 | 0.263 |

Tarla siniri segmentation modeli en iyi dogrulama IoU degeri:

| Metrik | Deger |
| --- | --- |
| IoU | 0.9528 |

## Test

```powershell
python -m pytest -q
```

## Notlar

- `weed_cluster` kesin yabanci ot teshisi olarak yorumlanmamalidir.
- `drydown` kesin kuraklik teshisi olarak yorumlanmamalidir.
- RGB goruntuler hassas bitki sagligi analizi icin sinirlidir.
- Daha ileri calismalarda NIR/multispektral veri ve problem siniflari icin
  segmentation yaklasimi kullanilabilir.
