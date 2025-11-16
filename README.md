# Analisis Karakter pada Karya Sastra

**Mini Project STKI (Sistem Temu Kembali Informasi)**  
Sistem ekstraksi dan analisis karakter dalam karya sastra berbahasa Inggris

---

## Tujuan Proyek

Project ini bertujuan untuk ekstraksi informasi dari karya sastra yang terdiri dari:

1. **Tokoh/Karakter** yang muncul dalam cerita
2. **Watak/Trait** dari setiap karakter
3. **Hubungan/Relasi** antar karakter
4. **Visualisasi** jaringan hubungan karakter

---

## Struktur Proyek

```
STKIProject/
│
├── data/                           # Data cerita
│   ├── raw/                        # Dokumen  (format .txt)
│   │   ├── owl_creek_bridge.txt
│   │   ├── the_gift_of_magi.txt
│   │   ├── the_tell_tale_heart.txt
│   │   └── the_yellow_wallpaper.txt
│   ├── processed/                  # Data hasil preprocessing (Output dihasilkan ketika menjalankan main.py)
│   └── results/                    # Hasil analisis (Output dihasilkan ketika menjalankan main.py)
│
├── src/                            # Source code utama sebagai proses ekstraksi
│   ├── __init__.py
│   ├── preprocessing.py            # Pembersihan & segmentasi teks
│   ├── ner_extraction.py           # Ekstraksi nama karakter
│   ├── trait_extraction.py         # Ekstraksi watak karakter
│   ├── relation_extraction.py      # Ekstraksi hubungan antar karakter
│   ├── name_normalizer.py          # Normalisasi variasi nama
│   └── utils.py                    # Utility & report generator
│
├── experiments/                    # Code eksperimen & testing
│   ├── __init__.py
│   ├── exp_01_ner_testing.py       # Eksperimen 1: Tes NER
│   ├── exp_02_trait_testing.py     # Eksperimen 2: Tes Ekstraksi watak
│   ├── exp_03_relation_testing.py  # Eksperimen 3: Tes Ekstraksi hubungan
│   ├── exp_04_full_pipeline.py     # Eksperimen 4: Full Pipeline
│   ├── debug_cooccurrence.py       # Debug co-occurrence detection
│   └── test_della_problem.py       # Debug character detection
│
├── outputs/                        # Hasil analisis & visualisasi
│   ├── reports/                    # Laporan JSON, Markdown, HTML
│   ├── visualizations/             # Grafik relasi karakter (PNG)
│   └── exp_*.json                  # Hasil eksperimen
│
├── main.py                         # Script utama untuk menjalankan sistem
├── requirements.txt                # Dependencies Python
└── README.md                       # Dokumentasi
```

---

## Eksperimen yang Dikerjakan

### **Eksperimen 1: Ekstraksi Nama**

`experiments/exp_01_ner_testing.py`

**Tujuan:** Menguji akurasi deteksi karakter dari cerita menggunakan Named Entity Recognition (NER).

**Metode:**

-   Ekstraksi menggunakan **spaCy NER**
-   **Pattern matching** untuk role-based characters (contoh: "The Old Man", "Narrator")
-   **Normalisasi nama** untuk merge variants (contoh: "Jim" → "James Dillingham Young")

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

📁 `experiments/exp_04_full_pipeline.py`

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

#### ** JSON Reports** (`outputs/reports/*.json`) ✅

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

#### ** Visualisasi Graf** (`outputs/visualizations/*.png`) ✅

Grafik jaringan hubungan karakter:

-   Node = Karakter
-   Edge = Hubungan (dengan label tipe relasi)
-   Thickness = Kekuatan hubungan
-   Color = Tingkat kepercayaan

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
