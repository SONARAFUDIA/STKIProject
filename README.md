```
STKIProject/
│
├── 📄 .gitignore                   # Git ignore rules
├── 📄 requirements.txt             # Python dependencies
├── 📄 README.md                    # Main documentation
├── 📄 main.py                      # CLI entry point
│
├── 📂 src/                         # Source code
│   ├── 📄 __init__.py
│   └── 📄 entity_extractor.py      # Core extraction logic
│
├── 📂 data/                        # Data files
│   ├── 📂 raw/                     # Original story files (.txt)
│   │   ├── 📄 the_gift_of_magi.txt
│   │   ├── 📄 owl_creek_bridge.txt
│   │   ├── 📄 the_tell_tale_heart.txt
│   │   └── 📄 the_yellow_wallpaper.txt
│   │
│   ├── 📂 processed/               # Preprocessed data (optional)
│   │
│   └── 📂 samples/                 # Test samples (optional)
│
├── 📂 outputs/                     # Generated outputs
│   ├── 📂 entities/                # Extracted entities (JSON, generated)
│   │
│   ├── 📂 reports/                 # Analysis reports (future, untuk watak dan relasi, generated)
│   │
│   └── 📂 visualizations/          # Graphs (future, generated)
│
├── 📂 tests/                       # Unit tests (future)
│   ├── 📄 __init__.py
│   └── 📄 test_extractor.py
│
├── 📂 configs/                     # Configuration files (optional future)
│   └── 📄 default.yaml
│
└── 📂 stki/                        # Virtual environment (auto-generated)
    └── (excluded from git)
```