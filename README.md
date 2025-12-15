# Analisis Karakter pada Karya Sastra

**Mini Project STKI (Sistem Temu Kembali Informasi)**  
Sistem ekstraksi dan analisis hubungan karakter dalam karya sastra berbahasa Inggris

---

## Tujuan Proyek

Project ini bertujuan untuk ekstraksi informasi dari karya sastra yang terdiri dari:

1. **Tokoh/Karakter** yang muncul dalam cerita
2. **Watak/Trait** dari setiap karakter
3. **Hubungan/Relasi** antar karakter
4. **Visualisasi** jaringan hubungan karakter

---

## Struktur Proyek
Updated at 23:51 - 15/12/2025

```
STKIProject/
│
├── data/                           # Data cerita (tidak berubah)
│   ├── raw/                        # Dokumen (format .txt)
│   ├── processed/
│   └── results/
│
├── src/                            # Source code
│   ├── __init__.py
│   ├── preprocessing.py            # Pembersihan & segmentasi teks (ENHANCED - tambah POS & n-gram)
│   ├── entity_extraction/          # NEW MODULE
│   │   ├── __init__.py
│   │   ├── base_extractor.py       # 🆕 Abstract base class
│   │   ├── method1_capitalization.py  # 🆕 Method 1
│   │   ├── method2_tfidf.py        # 🆕 Method 2
│   │   ├── method3_embeddings.py   # 🆕 Method 3
│   │   ├── ensemble_voter.py       # 🆕 Voting & fusion logic
│   │   └── entity_validator.py     # 🆕 Blacklist & validation
│   ├── trait_extraction.py         # (To be changed)
│   ├── relation_extraction.py      # (To be changed)
│   ├── name_normalizer.py          # ❌
│   └── utils.py                    # (To be changed)
│
├── experiments/                    # Experiments - UPDATED
│   ├── __init__.py
│   ├── exp_05_entity_extraction.py # (To be added)
│   ├── exp_06_method_comparison.py # (To be added)
│   ├── exp_07_ensemble_tuning.py   # (To be added)
│   └── ground_truth/               # (To be added)
│       ├── the_gift_of_magi_gt.json
│       ├── owl_creek_bridge_gt.json
│       └── ...
│
├── configs/                        # (To be added)
│   ├── default_config.yaml         # (To be added)
│   ├── method1_config.yaml         # (To be added)
│   ├── method2_config.yaml
│   └── method3_config.yaml
│
├── models/                         # (To be added)
│   └── embeddings/                 # (To be added)
│       └── all-MiniLM-L6-v2/
│
├── outputs/                        # Outputs
│   ├── reports/
│   ├── visualizations/
│   ├── metrics/                    # (To be changed)
│   └── debug/                      # (To be changed)
│
├── main.py                         # (To be changed)
├── requirements.txt                # (To be changed)
└── README.md                       # (To be changed)
```

---

## Eksperimen yang Dikerjakan

### **Eksperimen 1: Ekstraksi Nama**

`/src/entity_extraction`

**Tujuan:** Menguji akurasi deteksi karakter dari cerita

**Metode:**

Ekstraksi menggunakan:
1. Capitalization Mining (unsupervised)
2. TF-IDF Ranking (unsupervised)
3. BERTopic Clustering (unsupervised embeddings)
4. Voting mechanism dari 3 metode

**Hasil yang Diharapkan:**

-   Daftar karakter dengan jumlah sebutan masing-masing

**Output:**

```
✓ Characters found: 4
✓ Most mentioned: Della (45 mentions)

Characters:
  - Della: 45 mentions
  - Jim: 38 mentions
  - James Dillingham Young: 12 mentions
  - Sofronie: 4 mentions
```

---

### **Eksperimen 2: Ekstraksi Watak**
`experiments/exp_02_trait_testing.py`

**Tujuan:** Mengekstrak sifat/watak karakter dari konteks kalimat.

**Metode:**

-   **Adjective extraction**: mencari kata sifat di sekitar nama karakter
-   **Pattern matching**: pola seperti "CHARACTER is/was ADJECTIVE"
-   **Possessive descriptions**: ekstrak dari "CHARACTER's ADJECTIVE NOUN"
-   **Action-based inference**: deduksi watak dari tindakan karakter
-   **Sentiment analysis**: analisis sentimen konteks

**Klasifikasi Watak:**

-   **Positive**: kind, brave, loyal, generous, wise...
-   **Negative**: cruel, selfish, dishonest, wicked...
-   **Emotional**: sad, happy, angry, nervous, excited...
-   **Physical**: tall, beautiful, young, strong, pale...
-   **Behavioral**: cautious, aggressive, calm, impulsive...

**Output:**

```
Analyzing character: Della
  ✓ Total trait mentions: 23
  ✓ Unique traits: 12

Top 5 Traits:
  - beautiful: 5x
  - poor: 3x
  - loving: 3x
  - generous: 2x
  - emotional: 2x
```

---

### **Eksperimen 3: Ekstraksi Hubungan**

`experiments/exp_03_relation_testing.py`

**Tujuan:** Mendeteksi dan mengklasifikasi hubungan antar karakter.

**Metode:**

-   **Co-occurrence detection**: karakter yang muncul di kalimat yang sama
-   **Proximity detection**: karakter yang muncul dalam N kalimat berdekatan
-   **Pattern-based detection**: pola hubungan spesifik (e.g., "his wife", "married to")
-   **Possessive pronoun inference**: deduksi dari kata ganti kepemilikan

**Tipe Relasi yang Terdeteksi:**

-   **Family**: parent-child, siblings, married-couple, spouse, extended-family
-   **Romantic**: lovers, romantic-interest, husband-wife
-   **Social**: close-friends, acquaintances, companions, neighbors
-   **Professional**: colleagues, employer-employee, doctor-patient, customer-merchant
-   **Antagonistic**: enemies, rivals, victim-perpetrator, opposing-sides

**Output:**

```
Detected Relations:

1. Della ↔ Jim
   Primary Relation: lovers
   All Relations: lovers, married-couple
   Confidence: 0.73
   Strength: 0.79
   Co-occurrence: 1x
   Proximity: 65x

2. Narrator (I) ↔ The Old Man
   Primary Relation: victim-perpetrator
   Confidence: 0.98
   Strength: 0.99
```

**Output Visualisasi:** Graf jaringan hubungan karakter (NetworkX + Matplotlib)

---

### **Eksperimen 4: Full Pipeline**

`experiments/exp_04_full_pipeline.py`

**Tujuan:** Menjalankan seluruh pipeline analisis end-to-end.

**Proses:**

1. **Preprocessing** → Membersihkan teks & segmentasi kalimat
2. **Character Extraction** → Deteksi tokoh
3. **Trait Extraction** → Analisis watak
4. **Relation Extraction** → Deteksi hubungan
5. **Report Generation** → Laporan JSON, Markdown, HTML

**Output:**

-   Laporan lengkap per dokumen
-   Laporan gabungan semua dokumen
-   Statistik ringkasan

---

## Cara Menjalankan

### **1. Install Dependencies**

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### **2. Jalankan Mode Batch (Semua Dokumen)**

```bash
python main.py --mode batch
```

### **3. Jalankan Single Document**

```bash
python main.py --mode single --file data/raw/the_gift_of_magi.txt
```

### **4. Jalankan Eksperimen Spesifik**

```bash
# Eksperimen NER
python main.py --mode experiment --experiment ner

# Eksperimen Trait
python main.py --mode experiment --experiment trait

# Eksperimen Relation
python main.py --mode experiment --experiment relation

# Full Pipeline
python main.py --mode experiment --experiment full
```

---

## Hasil Ejstraksi

### **Output Files**

#### **JSON Reports** (`outputs/reports/*.json`) ✅

Hasil ekstraksi dalam format JSON untuk pemrosesan lebih lanjut.

```json
{
  "metadata": {
    "filename": "the_gift_of_magi.txt",
    "sentence_count": 156
  },
  "characters": {
    "statistics": {
      "total_characters": 4
    },
    "details": {
      "Della": 45,
      "Jim": 38
    }
  },
  "traits": { ... },
  "relations": { ... }
}
```

#### **Visualisasi Graf** (`outputs/visualizations/*.png`) ✅

Grafik jaringan hubungan karakter:

-   Node = Karakter
-   Edge = Hubungan (dengan label tipe relasi)
-   Thickness = Kekuatan hubungan
-   Color = Tingkat kepercayaan

#### **Markdown Reports** (`outputs/reports/*.md`)
Laporan human-readable dalam format Markdown.

#### **HTML Reports** (`outputs/reports/*.html`)
Laporan interaktif dengan styling untuk presentasi.

---

## 🔧 Library yang Digunakan

| Library        | Fungsi                                      |
| -------------- | ------------------------------------------- |
| **spaCy**      | Named Entity Recognition (NER), POS tagging |
| **NLTK**       | Tokenization, stopwords removal             |
| **TextBlob**   | Sentiment analysis                          |
| **NetworkX**   | Graph construction untuk relasi             |
| **Matplotlib** | Visualisasi graf relasi                     |
| **Pandas**     | Data manipulation (opsional)                |
| **NumPy**      | Operasi numerik                             |

---

## Kekurangan dan batasan:

### **1. Deteksi Nama**

-   Deteksi nama yang menggunakan gelar terkadang tidak muncul
-   **Generic references** ("the man", "the soldier") tidak di-track
-   Butuh story-specific role detection

### **2. Klasifikasi Hubungan**

-   Deteksi hubungan yang masih **generic**, menampilkan hubungan sederhana

### **3. Ekstraksi Watak**

-   Terbatas pada adjectives yang ada di keyword dictionary
-   Trait inference dari action verb masih sederhana
-   Butuh semantic analysis yang mendalam

### **4. Limitasi Bahasa**

-   Hanya dapat men-ekstraksi dokumen dalam Bahasa Inggris

---

## 📝 License

-

---

### **Cerita yang Digunakan:**

-   **"An Occurrence at Owl Creek Bridge"** - Ambrose Bierce (1890)
-   **"The Gift of the Magi"** - O. Henry (1905)
-   **"The Tell-Tale Heart"** - Edgar Allan Poe (1843)
-   **"The Yellow Wallpaper"** - Charlotte Perkins Gilman (1892)
