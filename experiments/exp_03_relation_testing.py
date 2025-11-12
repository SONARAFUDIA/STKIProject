import sys
import os

# Tambahkan root directory ke Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import TextPreprocessor
from src.ner_extraction import CharacterExtractor
from src.relation_extraction import RelationExtractor
import json
import networkx as nx
import matplotlib.pyplot as plt

def test_relation_extraction():
    """
    Eksperimen untuk testing ekstraksi hubungan
    """
    print("="*60)
    print("EKSPERIMEN 3: TESTING RELATION EXTRACTION")
    print("="*60)
    
    # Inisialisasi
    preprocessor = TextPreprocessor()
    char_extractor = CharacterExtractor()
    rel_extractor = RelationExtractor()
    
    # Test file
    filepath = 'data/raw/the_gift_of_magi.txt'
    
    print(f"\n📖 Processing: {filepath}")
    
    # Preprocessing
    preprocessed = preprocessor.preprocess_document(filepath)
    
    # Extract characters
    char_extraction = char_extractor.extract_characters(
        preprocessed['cleaned_text'],
        preprocessed['sentences']
    )
    
    print(f"✓ Characters found: {list(char_extraction['main_characters'].keys())}")
    
    # Extract relations
    relations = rel_extractor.extract_relations(
        char_extraction['main_characters'],
        preprocessed['sentences']
    )
    
    print(f"\n📊 Relation Analysis:")
    print(f"✓ Co-occurrence pairs: {len(relations['cooccurrence'])}")
    print(f"✓ Rule-based relations: {len(relations['rulebased'])}")
    print(f"✓ Merged relations: {len(relations['merged_relations'])}")
    
    # Tampilkan detail
    print("\n🔗 Detected Relations:")
    for rel in relations['merged_relations']:
        print(f"  {rel['character1']} ↔ {rel['character2']}")
        print(f"    - Co-occurrence: {rel['cooccurrence_count']}x")
        print(f"    - Types: {', '.join(rel['relation_types'])}")
        print(f"    - Strength: {rel['strength']:.2f}")
    
    # Visualisasi graph
    visualize_relation_graph(relations['relation_graph'], filepath)
    
    # Save results
    with open('outputs/exp_03_relation_results.json', 'w') as f:
        json.dump(relations, f, indent=2, default=str)
    
    print("\n✅ Results saved to: outputs/exp_03_relation_results.json")
    return relations

def visualize_relation_graph(graph_data, filepath):
    """
    Visualisasi graph relasi
    """
    G = nx.Graph()
    
    # Add nodes
    for node in graph_data['nodes']:
        G.add_node(node['id'])
    
    # Add edges
    for edge in graph_data['edges']:
        G.add_edge(edge['source'], edge['target'], weight=edge['weight'])
    
    # Plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', 
                           alpha=0.9, edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Draw edges dengan thickness berdasarkan weight
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=[w*5 for w in weights], alpha=0.5)
    
    plt.title(f"Character Relation Graph\n{filepath}", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    # Save
    output_path = f"outputs/visualizations/relation_graph_{filepath.split('/')[-1].replace('.txt', '.png')}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Graph saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    test_relation_extraction()