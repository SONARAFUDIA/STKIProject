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
    Eksperimen untuk testing ekstraksi hubungan - INDONESIAN VERSION
    """
    print("="*60)
    print("EKSPERIMEN 3: TESTING RELATION EXTRACTION (BAHASA INDONESIA)")
    print("="*60)
    
    # Inisialisasi
    preprocessor = TextPreprocessor()
    char_extractor = CharacterExtractor()
    rel_extractor = RelationExtractor()
    
    # Test files
    test_files = [
        'senja_di_ujung_kios.txt',
        'rapat_warung_yopi_yang_batal.txt',
        'aroma_kayu_cendana.txt',
        'asing_di_cermin_itu.txt',
        'garis_putus-putus.txt',
    ]
    
    all_results = {}
    
    for filename in test_files:
        filepath = os.path.join(PROJECT_ROOT, 'data/raw', filename)
        
        if not os.path.exists(filepath):
            print(f"\n⚠️  File tidak ditemukan: {filepath}")
            continue
        
        if os.path.getsize(filepath) == 0:
            print(f"\n⚠️  File kosong, dilewati: {filename}")
            continue
        
        print(f"\n{'='*60}")
        print(f"📖 Memproses: {filename}")
        print(f"{'='*60}")
        
        try:
            # Preprocessing
            print("\n[1/3] Preprocessing...")
            preprocessed = preprocessor.preprocess_document(filepath)
            print(f"  ✓ {preprocessed['sentence_count']} kalimat diekstrak")
            
            # Extract characters
            print("\n[2/3] Ekstraksi tokoh...")
            char_extraction = char_extractor.extract_characters(
                preprocessed['cleaned_text'],
                preprocessed['sentences'],
                min_mentions=2
            )
            
            if len(char_extraction['main_characters']) < 2:
                print(f"  ⚠️  Hanya {len(char_extraction['main_characters'])} tokoh ditemukan.")
                print("  ⚠️  Butuh minimal 2 tokoh untuk ekstraksi relasi. Melewati.")
                continue
            
            print(f"  ✓ {len(char_extraction['main_characters'])} tokoh ditemukan")
            print(f"  Tokoh: {', '.join(char_extraction['main_characters'].keys())}")
            
            # Extract relations
            print("\n[3/3] Ekstraksi relasi...")
            relations = rel_extractor.extract_relations(
                char_extraction['main_characters'],
                preprocessed['sentences']
            )
            
            print(f"\n📊 Analisis Relasi:")
            print(f"  ✓ Pasangan co-occurrence langsung: {len(relations.get('cooccurrence', {}))}")
            print(f"  ✓ Pasangan proximity: {len(relations.get('proximity_pairs', {}))}")
            print(f"  ✓ Relasi spesifik: {len(relations.get('specific_relations', []))}")
            print(f"  ✓ Relasi possessive: {len(relations.get('possessive_relations', []))}")
            print(f"  ✓ Relasi merged: {len(relations['merged_relations'])}")
            
            # Display details
            if relations['merged_relations']:
                print("\n🔗 Relasi yang Terdeteksi:")
                for i, rel in enumerate(relations['merged_relations'][:10], 1):
                    print(f"\n  {i}. {rel['character1']} ↔ {rel['character2']}")
                    print(f"     Relasi Utama: {rel['primary_relation']}")
                    print(f"     Semua Relasi: {', '.join(rel['all_relations'])}")
                    print(f"     Kepercayaan: {rel['confidence']:.2f}")
                    print(f"     Kekuatan: {rel['strength']:.2f}")
                    print(f"     Co-occurrence: {rel['cooccurrence_count']}x")
                    print(f"     Proximity: {rel['proximity_count']}x")
                    print(f"     Sumber: {rel['source']}")
            else:
                print("\n  ⚠️  Tidak ada relasi terdeteksi")
            
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
            print(f"\n❌ Kesalahan memproses {filename}: {str(e)}")
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
        print(f"✅ Hasil tersimpan di: {output_file}")
        print("="*60)
        
        # Enhanced Summary
        print("\n📊 RINGKASAN LENGKAP:")
        print("="*60)
        for filename, results in all_results.items():
            summary = results['summary']
            print(f"\n📖 {filename}:")
            print(f"  Tokoh: {summary['total_characters']}")
            print(f"  Relasi terdeteksi: {summary['total_relations']}")
            print(f"  Pasangan proximity: {summary.get('proximity_pairs', 0)}")
            print(f"  Relasi spesifik: {summary.get('specific_relations', 0)}")
            
            if summary['total_relations'] > 0:
                # Show top relation dengan details
                top_rel = results['relations'][0]
                print(f"\n  🔗 Relasi Teratas:")
                print(f"     {top_rel['character1']} ↔ {top_rel['character2']}")
                print(f"     Tipe: {top_rel['primary_relation']}")
                print(f"     Kepercayaan: {top_rel['confidence']:.2f}")
                print(f"     Kekuatan: {top_rel['strength']:.2f}")
                
                # Show all detected relation types
                all_relation_types = set()
                for rel in results['relations']:
                    all_relation_types.update(rel['all_relations'])
                
                if all_relation_types:
                    print(f"\n  📋 Semua Tipe Relasi yang Terdeteksi:")
                    for rel_type in sorted(all_relation_types):
                        count = sum(1 for rel in results['relations'] 
                                   if rel_type in rel['all_relations'])
                        print(f"     - {rel_type}: {count}x")
        
        print("\n" + "="*60)
        print("✅ ANALISIS SELESAI!")
        print("="*60)
    else:
        print("\n⚠️  Tidak ada hasil untuk disimpan")

    return all_results

def visualize_relation_graph(graph_data, filename):
    """
    Visualisasi graph relasi - Indonesian version
    """
    try:
        if not graph_data['nodes'] or not graph_data['edges']:
            print(f"\n  ⚠️  Tidak ada relasi untuk divisualisasikan: {filename}")
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
        
        # Draw nodes dengan warna berbeda
        node_colors = []
        for node in G.nodes():
            if 'narator' in node.lower() or 'aku' in node.lower():
                node_colors.append('#FF6B6B')  # Red untuk narrator
            elif any(honor in node.lower() for honor in ['pak', 'bu', 'mas', 'mbak']):
                node_colors.append('#4ECDC4')  # Teal untuk dengan gelar
            else:
                node_colors.append('#95E1D3')  # Light green untuk nama biasa
        
        nx.draw_networkx_nodes(
            G, pos, 
            node_size=4000, 
            node_color=node_colors,
            alpha=0.9, 
            edgecolors='black', 
            linewidths=2.5
        )
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_family='sans-serif')
        
        # Draw edges
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        confidences = [G[u][v].get('confidence', 0.5) for u, v in edges]
        
        # Normalize weights
        max_weight = max(weights) if weights else 1
        normalized_weights = [w/max_weight * 6 + 1 for w in weights]
        
        # Color edges by confidence
        edge_colors = []
        for conf in confidences:
            if conf >= 0.8:
                edge_colors.append('#2ECC71')  # Green untuk high confidence
            elif conf >= 0.6:
                edge_colors.append('#F39C12')  # Orange untuk medium
            else:
                edge_colors.append('#E74C3C')  # Red untuk low
        
        nx.draw_networkx_edges(
            G, pos, 
            width=normalized_weights, 
            alpha=0.7,
            edge_color=edge_colors
        )
        
        # Add edge labels
        edge_labels = {}
        for edge_data in graph_data['edges']:
            edge_key = (edge_data['source'], edge_data['target'])
            relation = edge_data['relation']
            confidence = edge_data.get('confidence', 0)
            
            # Shorten relation name
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
        
        # Title
        plt.title(
            f"Graf Hubungan Tokoh: {filename}\n"
            f"({len(G.nodes())} tokoh, {len(G.edges())} relasi)",
            fontsize=16, 
            fontweight='bold',
            pad=20
        )
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#95E1D3', edgecolor='black', label='Nama Tokoh'),
            Patch(facecolor='#4ECDC4', edgecolor='black', label='Tokoh dengan Gelar'),
            Patch(facecolor='#FF6B6B', edgecolor='black', label='Narator'),
            Patch(facecolor='#2ECC71', label='Kepercayaan Tinggi (≥0.8)'),
            Patch(facecolor='#F39C12', label='Kepercayaan Sedang (0.6-0.8)'),
            Patch(facecolor='#E74C3C', label='Kepercayaan Rendah (<0.6)')
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
        print(f"  ✓ Graf tersimpan: {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"\n  ⚠️  Tidak bisa membuat visualisasi untuk {filename}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_relation_extraction()