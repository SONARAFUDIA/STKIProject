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
        
        md_content = f"""# Character Analysis Report: {doc_name}

## Document Information
- **Filename**: {doc_name}
- **Processed**: {analysis_result['metadata']['processed_at']}
- **Total Sentences**: {analysis_result['metadata']['sentence_count']}

## Characters Detected

### Main Characters
Total: {analysis_result['characters']['statistics']['total_characters']}

"""
        
        # List characters with mention count
        for char, count in analysis_result['characters']['details'].items():
            md_content += f"- **{char}**: {count} mentions\n"
        
        # Character traits
        md_content += "\n## Character Traits\n\n"
        for char, traits in analysis_result['traits'].items():
            md_content += f"### {char}\n\n"
            
            if traits['classified_traits']:
                for category, trait_list in traits['classified_traits'].items():
                    if trait_list:
                        unique_traits = list(set(trait_list))
                        md_content += f"**{category.title()}**: {', '.join(unique_traits)}\n\n"
            
            # Top traits
            if traits['trait_frequency']:
                top_traits = sorted(traits['trait_frequency'].items(), 
                                   key=lambda x: x[1], reverse=True)[:5]
                md_content += "**Most Frequent Traits**:\n"
                for trait, freq in top_traits:
                    md_content += f"- {trait} ({freq}x)\n"
            
            md_content += "\n"
        
        # Relations
        md_content += "## Character Relations\n\n"
        md_content += f"Total Relations: {analysis_result['relations']['summary']['total_relations']}\n\n"
        
        for rel in analysis_result['relations']['details']:
            md_content += f"### {rel['character1']} ↔ {rel['character2']}\n"
            md_content += f"- **Co-occurrence**: {rel['cooccurrence_count']} times\n"
            md_content += f"- **Relation Types**: {', '.join(rel['relation_types'])}\n"
            md_content += f"- **Strength**: {rel['strength']:.2f}\n\n"
        
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
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Character Analysis: {doc_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
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
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #2ecc71;
        }}
        .trait-badge {{
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 5px 10px;
            margin: 3px;
            border-radius: 3px;
            font-size: 0.9em;
        }}
        .relation-card {{
            background-color: #fff3cd;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 Character Analysis Report</h1>
        <h2>{doc_name}</h2>
        
        <div class="stat-box">
            <strong>Total Sentences:</strong> {analysis_result['metadata']['sentence_count']}
        </div>
        <div class="stat-box">
            <strong>Characters Found:</strong> {analysis_result['characters']['statistics']['total_characters']}
        </div>
        <div class="stat-box">
            <strong>Relations Found:</strong> {analysis_result['relations']['summary']['total_relations']}
        </div>
        
        <h2>🎭 Main Characters</h2>
"""
        
        # Characters
        for char, count in analysis_result['characters']['details'].items():
            html_content += f"""
        <div class="character-card">
            <h3>{char}</h3>
            <p><strong>Mentions:</strong> {count}</p>
"""
            
            # Traits
            if char in analysis_result['traits']:
                traits = analysis_result['traits'][char]
                html_content += "<p><strong>Traits:</strong><br>"
                
                for category, trait_list in traits['classified_traits'].items():
                    if trait_list:
                        for trait in set(trait_list):
                            html_content += f'<span class="trait-badge">{trait}</span>'
                
                html_content += "</p>"
            
            html_content += "</div>"
        
        # Relations
        html_content += "<h2>🔗 Character Relations</h2>"
        
        for rel in analysis_result['relations']['details']:
            html_content += f"""
        <div class="relation-card">
            <h4>{rel['character1']} ↔ {rel['character2']}</h4>
            <p>
                <strong>Co-occurrence:</strong> {rel['cooccurrence_count']} times<br>
                <strong>Relation Types:</strong> {', '.join(rel['relation_types'])}<br>
                <strong>Strength:</strong> {rel['strength']:.2f}
            </p>
        </div>
"""
        
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
    print(f"✅ Data saved: {filepath}")

def load_processed_data(filepath):
    """
    Utility untuk memuat data yang sudah diproses
    """
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None