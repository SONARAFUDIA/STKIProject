import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import TextPreprocessor
from src.ner_extraction import CharacterExtractor
from src.relation_extraction import RelationExtractor
import json
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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
    filepath = os.path.join(PROJECT_ROOT, 'data/raw/the_gift_of_magi.txt')
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    
    print(f"\n📖 Processing: {os.path.basename(filepath)}")
    
    try:
        # Preprocessing
        print("\n[1/3] Preprocessing...")
        preprocessed = preprocessor.preprocess_document(filepath)
        print(f"  ✓ {preprocessed['sentence_count']} sentences extracted")
        
        # Extract characters
        print("\n[2/3] Extracting characters...")
        char_extraction = char_extractor.extract_characters(
            preprocessed['cleaned_text'],
            preprocessed['sentences'],
            min_mentions=2
        )
        
        print(f"  ✓ {len(char_extraction['main_characters'])} characters found")
        print(f"  Characters: {', '.join(char_extraction['main_characters'].keys())}")
        
        # Extract relations
        print("\n[3/3] Extracting relations...")
        relations = rel_extractor.extract_relations(
            char_extraction['main_characters'],
            preprocessed['sentences']
        )
        
        print(f"\n📊 Relation Analysis:")
        print(f"  ✓ Co-occurrence pairs: {len(relations['cooccurrence'])}")
        print(f"  ✓ Rule-based relations: {len(relations['rulebased'])}")
        print(f"  ✓ Merged relations: {len(relations['merged_relations'])}")
        
        # Tampilkan detail
        print("\n🔗 Detected Relations:")
        for i, rel in enumerate(relations['merged_relations'][:10], 1):  # Show top 10
            print(f"\n  {i}. {rel['character1']} ↔ {rel['character2']}")
            print(f"     Co-occurrence: {rel['cooccurrence_count']}x")
            print(f"     Types: {', '.join(rel['relation_types'])}")
            print(f"     Strength: {rel['strength']:.2f}")
        
        # Visualisasi graph
        if len(relations['merged_relations']) > 0:
            visualize_relation_graph(relations['relation_graph'], os.path.basename(filepath))
        
        # Prepare results for JSON - CONVERT TUPLE KEYS TO STRINGS
        cooccurrence_serializable = {}
        for key, value in relations['cooccurrence'].items():
            # Convert tuple key to string
            if isinstance(key, tuple):
                key_str = f"{key[0]} <-> {key[1]}"
            else:
                key_str = str(key)
            
            # Only include count, not sentences (too verbose)
            cooccurrence_serializable[key_str] = {
                'count': value['count'],
                'sentence_count': len(value.get('sentences', []))
            }
        
        json_results = {
            'summary': {
                'total_cooccurrence_pairs': len(relations['cooccurrence']),
                'total_rulebased': len(relations['rulebased']),
                'total_merged': len(relations['merged_relations'])
            },
            'cooccurrence': cooccurrence_serializable,
            'relations': relations['merged_relations'],
            'graph': relations['relation_graph']
        }
        
        # Save results
        output_dir = os.path.join(PROJECT_ROOT, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, 'exp_03_relation_results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*60)
        print(f"✅ Results saved to: {output_file}")
        print("="*60)
        
        return relations
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def visualize_relation_graph(graph_data, filename):
    """
    Visualisasi graph relasi
    """
    try:
        if not graph_data['nodes'] or not graph_data['edges']:
            print("\n⚠️  No relations to visualize")
            return
        
        G = nx.Graph()
        
        # Add nodes
        for node in graph_data['nodes']:
            G.add_node(node['id'])
        
        # Add edges
        for edge in graph_data['edges']:
            G.add_edge(edge['source'], edge['target'], weight=edge['weight'])
        
        # Plot
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=3500, node_color='lightblue', 
                               alpha=0.9, edgecolors='black', linewidths=2)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
        
        # Draw edges with thickness based on weight
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w*5 for w in weights], alpha=0.5)
        
        # Add edge labels (relation types)
        edge_labels = {}
        for edge_data in graph_data['edges']:
            edge_key = (edge_data['source'], edge_data['target'])
            types = ', '.join(edge_data['types'][:2])  # Show first 2 types
            edge_labels[edge_key] = types
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7)
        
        plt.title(f"Character Relation Graph\n{filename}", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Save
        output_dir = os.path.join(PROJECT_ROOT, 'outputs/visualizations')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"relation_graph_{filename.replace('.txt', '.png')}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Graph visualization saved to: {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"\n⚠️  Warning: Could not create visualization: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_relation_extraction()