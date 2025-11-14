import json
import os
from collections import Counter

class ReportGenerator:
    """
    Generator untuk laporan analisis
    """
    
    @staticmethod
    def generate_markdown_report(analysis_result, output_path):
        """
        Generate laporan dalam format Markdown
        """
        doc_name = analysis_result['metadata']['filename']
        
        md_content = f"""# Laporan Analisis Karakter: {doc_name}

## Informasi Dokumen
- **Nama File**: {doc_name}
- **Diproses Pada**: {analysis_result['metadata']['processed_at']}
- **Total Kalimat**: {analysis_result['metadata']['sentence_count']}

## Tokoh yang Terdeteksi

### Tokoh Utama
Total: {analysis_result['characters']['statistics']['total_characters']}

"""
        
        # List karakter dengan jumlah sebutan
        for char, count in analysis_result['characters']['details'].items():
            md_content += f"- **{char}**: {count} sebutan\n"
        
        # Watak tokoh
        md_content += "\n## Watak Tokoh\n\n"
        for char, traits in analysis_result['traits'].items():
            md_content += f"### {char}\n\n"
            
            if traits['classified_traits']:
                for category, trait_list in traits['classified_traits'].items():
                    if trait_list:
                        unique_traits = list(set(trait_list))
                        category_indo = {
                            'positive': 'Positif',
                            'negative': 'Negatif',
                            'emotional': 'Emosional',
                            'physical': 'Fisik',
                            'behavioral': 'Perilaku',
                            'other': 'Lainnya'
                        }.get(category, category.title())
                        
                        md_content += f"**{category_indo}**: {', '.join(unique_traits)}\n\n"
            
            # Top traits
            if traits['trait_frequency']:
                top_traits = sorted(traits['trait_frequency'].items(), 
                                   key=lambda x: x[1], reverse=True)[:5]
                md_content += "**Watak yang Paling Sering Muncul**:\n"
                for trait, freq in top_traits:
                    md_content += f"- {trait} ({freq}x)\n"
            
            md_content += "\n"
        
        # Hubungan antar tokoh
        md_content += "## Hubungan Antar Tokoh\n\n"
        md_content += f"Total Hubungan: {analysis_result['relations']['summary']['total_relations']}\n\n"
        
        for rel in analysis_result['relations']['details']:
            md_content += f"### {rel['character1']} ↔ {rel['character2']}\n"
            md_content += f"- **Kemunculan Bersama**: {rel['cooccurrence_count']} kali\n"
            md_content += f"- **Jenis Hubungan**: {', '.join(rel['relation_types'])}\n"
            md_content += f"- **Kekuatan**: {rel['strength']:.2f}\n\n"
        
        # Save
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return output_path
    
    @staticmethod
    def generate_html_report(analysis_result, output_path):
        """
        Generate laporan dalam format HTML
        """
        doc_name = analysis_result['metadata']['filename']
        
        html_content = f"""<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analisis Karakter: {doc_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        h3 {{
            color: #546e7a;
            margin-top: 20px;
        }}
        .stat-box {{
            display: inline-block;
            background-color: #ecf0f1;
            padding: 15px 25px;
            margin: 10px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .character-card {{
            background-color: #f8f9fa;
            padding: 20px;
            margin: 15px 0;
            border-radius: 8px;
            border-left: 5px solid #2ecc71;
        }}
        .trait-badge {{
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 5px 12px;
            margin: 3px;
            border-radius: 15px;
            font-size: 0.9em;
        }}
        .trait-badge.positive {{
            background-color: #2ecc71;
        }}
        .trait-badge.negative {{
            background-color: #e74c3c;
        }}
        .trait-badge.emotional {{
            background-color: #9b59b6;
        }}
        .trait-badge.physical {{
            background-color: #f39c12;
        }}
        .trait-badge.behavioral {{
            background-color: #1abc9c;
        }}
        .relation-card {{
            background-color: #fff3cd;
            padding: 20px;
            margin: 15px 0;
            border-radius: 8px;
            border-left: 5px solid #ffc107;
        }}
        .strength-bar {{
            width: 100%;
            height: 20px;
            background-color: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 10px;
        }}
        .strength-fill {{
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            transition: width 0.3s ease;
        }}
        .metadata {{
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .no-data {{
            color: #95a5a6;
            font-style: italic;
            padding: 20px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 Laporan Analisis Karakter</h1>
        <h2 style="border: none; color: #7f8c8d;">{doc_name}</h2>
        
        <div class="metadata">
            <strong>📅 Diproses:</strong> {analysis_result['metadata']['processed_at']}<br>
            <strong>📄 Total Kalimat:</strong> {analysis_result['metadata']['sentence_count']}
        </div>
        
        <div class="stat-box">
            <strong>👥 Tokoh Ditemukan:</strong> {analysis_result['characters']['statistics']['total_characters']}
        </div>
        <div class="stat-box">
            <strong>🔗 Hubungan Ditemukan:</strong> {analysis_result['relations']['summary']['total_relations']}
        </div>
        
        <h2>🎭 Tokoh Utama</h2>
"""
        
        # Tokoh-tokoh
        for char, count in analysis_result['characters']['details'].items():
            html_content += f"""
        <div class="character-card">
            <h3>{char}</h3>
            <p><strong>Jumlah Sebutan:</strong> {count}x</p>
"""
            
            # Watak
            if char in analysis_result['traits']:
                traits = analysis_result['traits'][char]
                html_content += "<p><strong>Watak yang Teridentifikasi:</strong><br>"
                
                category_map = {
                    'positive': 'Positif',
                    'negative': 'Negatif',
                    'emotional': 'Emosional',
                    'physical': 'Fisik',
                    'behavioral': 'Perilaku',
                    'other': 'Lainnya'
                }
                
                for category, trait_list in traits['classified_traits'].items():
                    if trait_list:
                        category_indo = category_map.get(category, category.title())
                        html_content += f"<div style='margin: 10px 0;'><strong>{category_indo}:</strong><br>"
                        for trait in set(trait_list):
                            html_content += f'<span class="trait-badge {category}">{trait}</span>'
                        html_content += "</div>"
                
                html_content += "</p>"
            
            html_content += "</div>"
        
        # Hubungan
        html_content += "<h2>🔗 Hubungan Antar Tokoh</h2>"
        
        if analysis_result['relations']['details']:
            for rel in analysis_result['relations']['details']:
                strength_percent = int(rel['strength'] * 100)
                html_content += f"""
        <div class="relation-card">
            <h4>{rel['character1']} ↔ {rel['character2']}</h4>
            <p>
                <strong>Kemunculan Bersama:</strong> {rel['cooccurrence_count']} kali<br>
                <strong>Jenis Hubungan:</strong> {', '.join(rel['relation_types'])}<br>
                <strong>Kepercayaan:</strong> {rel['confidence']:.2f}<br>
                <strong>Kekuatan Hubungan:</strong> {rel['strength']:.2f}
            </p>
            <div class="strength-bar">
                <div class="strength-fill" style="width: {strength_percent}%"></div>
            </div>
        </div>
"""
        else:
            html_content += '<div class="no-data">Tidak ada hubungan antar tokoh yang terdeteksi.</div>'
        
        html_content += """
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path

def save_processed_data(data, filepath):
    """
    Utility untuk menyimpan data yang sudah diproses
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Data tersimpan: {filepath}")

def load_processed_data(filepath):
    """
    Utility untuk memuat data yang sudah diproses
    """
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None