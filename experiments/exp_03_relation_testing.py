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
    Eksperimen untuk testing ekstraksi hubungan - UPDATED VERSION
    """
    print("="*60)
    print("EKSPERIMEN 3: TESTING RELATION EXTRACTION (ENHANCED)")
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
            print(f"  ✓ Direct co-occurrence pairs: {len(relations.get('cooccurrence', {}))}")
            print(f"  ✓ Proximity pairs: {len(relations.get('proximity_pairs', {}))}")
            print(f"  ✓ Specific relations: {len(relations.get('specific_relations', []))}")
            print(f"  ✓ Possessive relations: {len(relations.get('possessive_relations', []))}")
            print(f"  ✓ Merged relations: {len(relations['merged_relations'])}")
            
            # Display details
            if relations['merged_relations']:
                print("\n🔗 Detected Relations:")
                for i, rel in enumerate(relations['merged_relations'][:10], 1):
                    print(f"\n  {i}. {rel['character1']} ↔ {rel['character2']}")
                    print(f"     Primary Relation: {rel['primary_relation']}")
                    print(f"     All Relations: {', '.join(rel['all_relations'])}")
                    print(f"     Confidence: {rel['confidence']:.2f}")
                    print(f"     Strength: {rel['strength']:.2f}")
                    print(f"     Co-occurrence: {rel['cooccurrence_count']}x")
                    print(f"     Proximity: {rel['proximity_count']}x")
                    print(f"     Source: {rel['source']}")
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
                    'characters': list(char_extraction['main_characters'].keys()),
                    'proximity_pairs': len(relations.get('proximity_pairs', {})),
                    'specific_relations': len(relations.get('specific_relations', []))
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
        
        # Enhanced Summary
        print("\n📊 ENHANCED SUMMARY:")
        print("="*60)
        for filename, results in all_results.items():
            summary = results['summary']
            print(f"\n📖 {filename}:")
            print(f"  Characters: {summary['total_characters']}")
            print(f"  Relations detected: {summary['total_relations']}")
            print(f"  Proximity pairs: {summary.get('proximity_pairs', 0)}")
            print(f"  Specific relations: {summary.get('specific_relations', 0)}")
            
            if summary['total_relations'] > 0:
                # Show top relation with details
                top_rel = results['relations'][0]
                print(f"\n  🔗 Top Relation:")
                print(f"     {top_rel['character1']} ↔ {top_rel['character2']}")
                print(f"     Type: {top_rel['primary_relation']}")
                print(f"     Confidence: {top_rel['confidence']:.2f}")
                print(f"     Strength: {top_rel['strength']:.2f}")
                
                # Show all detected relation types
                all_relation_types = set()
                for rel in results['relations']:
                    all_relation_types.update(rel['all_relations'])
                
                if all_relation_types:
                    print(f"\n  📋 All Relation Types Detected:")
                    for rel_type in sorted(all_relation_types):
                        count = sum(1 for rel in results['relations'] 
                                   if rel_type in rel['all_relations'])
                        print(f"     - {rel_type}: {count}x")
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print("="*60)
    else:
        print("\n⚠️  No results to save")

    return all_results

def visualize_relation_graph(graph_data, filename):
    """
    Visualisasi graph relasi - ENHANCED VERSION
    """
    try:
        if not graph_data['nodes'] or not graph_data['edges']:
            print(f"\n  ⚠️  No relations to visualize for {filename}")
            return
        
        G = nx.Graph()
        
        # Add nodes
        for node in graph_data['nodes']:
            G.add_node(node['id'])
        
        # Add edges with attributes
        for edge in graph_data['edges']:
            G.add_edge(
                edge['source'], 
                edge['target'], 
                weight=edge['weight'],
                relation=edge['relation'],
                types=edge.get('types', []),
                confidence=edge.get('confidence', 0)
            )
        
        # Create figure
        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
        
        # Draw nodes with different colors based on character type
        node_colors = []
        for node in G.nodes():
            if 'narrator' in node.lower():
                node_colors.append('#FF6B6B')  # Red for narrator
            elif 'the ' in node.lower():
                node_colors.append('#4ECDC4')  # Teal for role-based
            else:
                node_colors.append('#95E1D3')  # Light green for named characters
        
        nx.draw_networkx_nodes(
            G, pos, 
            node_size=4000, 
            node_color=node_colors,
            alpha=0.9, 
            edgecolors='black', 
            linewidths=2.5
        )
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        # Draw edges with varying thickness and color
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        confidences = [G[u][v].get('confidence', 0.5) for u, v in edges]
        
        # Normalize weights for visual display
        max_weight = max(weights) if weights else 1
        normalized_weights = [w/max_weight * 6 + 1 for w in weights]
        
        # Color edges by confidence
        edge_colors = []
        for conf in confidences:
            if conf >= 0.8:
                edge_colors.append('#2ECC71')  # Green for high confidence
            elif conf >= 0.6:
                edge_colors.append('#F39C12')  # Orange for medium
            else:
                edge_colors.append('#E74C3C')  # Red for low
        
        nx.draw_networkx_edges(
            G, pos, 
            width=normalized_weights, 
            alpha=0.7,
            edge_color=edge_colors
        )
        
        # Add edge labels with relation type
        edge_labels = {}
        for edge_data in graph_data['edges']:
            edge_key = (edge_data['source'], edge_data['target'])
            relation = edge_data['relation']
            confidence = edge_data.get('confidence', 0)
            
            # Shorten relation name if too long
            if len(relation) > 15:
                relation = relation[:12] + '...'
            
            label = f"{relation}\n({confidence:.2f})"
            edge_labels[edge_key] = label
        
        nx.draw_networkx_edge_labels(
            G, pos, 
            edge_labels, 
            font_size=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )
        
        # Title with enhanced info
        plt.title(
            f"Character Relation Graph: {filename}\n"
            f"({len(G.nodes())} characters, {len(G.edges())} relations)",
            fontsize=16, 
            fontweight='bold',
            pad=20
        )
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#95E1D3', edgecolor='black', label='Named Character'),
            Patch(facecolor='#4ECDC4', edgecolor='black', label='Role-based Character'),
            Patch(facecolor='#FF6B6B', edgecolor='black', label='Narrator'),
            Patch(facecolor='#2ECC71', label='High Confidence (≥0.8)'),
            Patch(facecolor='#F39C12', label='Medium Confidence (0.6-0.8)'),
            Patch(facecolor='#E74C3C', label='Low Confidence (<0.6)')
        ]
        plt.legend(handles=legend_elements, loc='upper left', fontsize=9)
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save
        output_dir = os.path.join(PROJECT_ROOT, 'outputs/visualizations')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(
            output_dir, 
            f"relation_graph_{filename.replace('.txt', '.png')}"
        )
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Graph saved: {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"\n  ⚠️  Warning: Could not create visualization for {filename}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_relation_extraction()