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
    Eksperimen untuk testing ekstraksi hubungan - MULTIPLE FILES
    """
    print("="*60)
    print("EKSPERIMEN 3: TESTING RELATION EXTRACTION")
    print("="*60)
    
    # Inisialisasi
    preprocessor = TextPreprocessor()
    char_extractor = CharacterExtractor()
    rel_extractor = RelationExtractor()
    
    # Test files
    test_files = [
        'owl_creek_bridge.txt',
        'the_gift_of_magi.txt',
        'the_tell_tale_heart.txt',
        'the_yellow_wallpaper.txt',
    ]
    
    all_results = {}
    
    for filename in test_files:
        filepath = os.path.join(PROJECT_ROOT, 'data/raw', filename)
        
        if not os.path.exists(filepath):
            print(f"\n⚠️  File not found: {filepath}")
            continue
        
        print(f"\n{'='*60}")
        print(f"📖 Processing: {filename}")
        print(f"{'='*60}")
        
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
            
            if len(char_extraction['main_characters']) < 2:
                print(f"  ⚠️  Only {len(char_extraction['main_characters'])} character(s) found.")
                print("  ⚠️  Need at least 2 characters for relation extraction. Skipping.")
                continue
            
            print(f"  ✓ {len(char_extraction['main_characters'])} characters found")
            print(f"  Characters: {', '.join(char_extraction['main_characters'].keys())}")
            
            # Extract relations
            print("\n[3/3] Extracting relations...")
            relations = rel_extractor.extract_relations(
                char_extraction['main_characters'],
                preprocessed['sentences']
            )
            
            print(f"\n📊 Relation Analysis:")
            print(f"  ✓ Direct co-occurrence pairs: {len(relations.get('direct_cooccurrence', {}))}")
            print(f"  ✓ Proximity relations: {len(relations.get('proximity', {}))}")
            print(f"  ✓ Semantic relations: {len(relations.get('semantic', []))}")
            print(f"  ✓ Merged relations: {len(relations['merged_relations'])}")
            
            # Display details
            if relations['merged_relations']:
                print("\n🔗 Detected Relations:")
                for i, rel in enumerate(relations['merged_relations'][:10], 1):
                    print(f"\n  {i}. {rel['character1']} ↔ {rel['character2']}")
                    print(f"     Co-occurrence: {rel['cooccurrence_count']}x")
                    print(f"     Types: {', '.join(rel['relation_types'])}")
                    print(f"     Strength: {rel['strength']:.2f}")
            else:
                print("\n  ⚠️  No relations detected")
            
            # Visualize graph
            if len(relations['merged_relations']) > 0:
                visualize_relation_graph(relations['relation_graph'], filename)
            
            # Prepare results for JSON
            json_results = {
                'summary': {
                    'total_characters': len(char_extraction['main_characters']),
                    'total_relations': len(relations['merged_relations']),
                    'characters': list(char_extraction['main_characters'].keys())
                },
                'relations': relations['merged_relations'],
                'graph': relations['relation_graph']
            }
            
            all_results[filename] = json_results
            
        except Exception as e:
            print(f"\n❌ Error processing {filename}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save results
    if all_results:
        output_dir = os.path.join(PROJECT_ROOT, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, 'exp_03_relation_results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*60)
        print(f"✅ Results saved to: {output_file}")
        print("="*60)
        
        # Summary - SINGLE PRINT ONLY
        print("\n📊 SUMMARY:")
        for filename, results in all_results.items():
            summary = results['summary']
            print(f"\n  {filename}:")
            print(f"    Characters: {summary['total_characters']}")
            print(f"    Relations: {summary['total_relations']}")
            if summary['total_relations'] > 0:
                top_rel = results['relations'][0]
                print(f"    Top: {top_rel['character1']} ↔ {top_rel['character2']} (strength: {top_rel['strength']:.2f})")
    else:
        print("\n⚠️  No results to save")

    return all_results

def visualize_relation_graph(graph_data, filename):
    """
    Visualisasi graph relasi
    """
    try:
        if not graph_data['nodes'] or not graph_data['edges']:
            print(f"\n  ⚠️  No relations to visualize for {filename}")
            return
        
        G = nx.Graph()
        
        # Add nodes
        for node in graph_data['nodes']:
            G.add_node(node['id'])
        
        # Add edges
        for edge in graph_data['edges']:
            G.add_edge(edge['source'], edge['target'], 
                      weight=edge['weight'],
                      types=edge.get('types', []))
        
        # Plot
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=3500, node_color='lightblue', 
                               alpha=0.9, edgecolors='black', linewidths=2)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
        
        # Draw edges with thickness based on weight
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        normalized_weights = [w/max_weight * 5 for w in weights]
        nx.draw_networkx_edges(G, pos, width=normalized_weights, alpha=0.6)
        
        # Add edge labels
        edge_labels = {}
        for edge_data in graph_data['edges']:
            edge_key = (edge_data['source'], edge_data['target'])
            types = edge_data.get('types', ['unknown'])
            # Show first 2 types
            label = ', '.join(types[:2])
            edge_labels[edge_key] = label
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7)
        
        plt.title(f"Character Relation Graph\n{filename}", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Save
        output_dir = os.path.join(PROJECT_ROOT, 'outputs/visualizations')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"relation_graph_{filename.replace('.txt', '.png')}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Graph saved: {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"\n  ⚠️  Warning: Could not create visualization for {filename}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_relation_extraction()