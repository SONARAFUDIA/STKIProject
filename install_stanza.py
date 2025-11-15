"""
Script untuk install dan setup Stanza Indonesian model
Run: python install_stanza.py
"""

import stanza
import os

def install_stanza_indonesian():
    """
    Download dan install Stanza Indonesian model
    """
    print("="*70)
    print("INSTALLING STANZA INDONESIAN MODEL")
    print("="*70)
    
    print("\n[1/2] Downloading Indonesian model...")
    print("This may take a few minutes (300-400MB download)...\n")
    
    try:
        # Download Indonesian model
        # processors: tokenize, mwt, pos, lemma, depparse, ner
        stanza.download('id', verbose=True)
        
        print("\n✅ Model downloaded successfully!")
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("Please check your internet connection and try again.")
        return False
    
    print("\n[2/2] Testing installation...")
    
    try:
        # Test the model (WITHOUT NER - not available for Indonesian)
        nlp = stanza.Pipeline('id', processors='tokenize,pos', verbose=False)
        
        # Test sentence
        test_text = "Pak Suroto adalah seorang tukang sol sepatu di Jakarta."
        doc = nlp(test_text)
        
        print("\n✓ Test successful!")
        print(f"\n  Input: {test_text}")
        print(f"  Tokens: {len(doc.sentences[0].tokens)}")
        
        # Show entities
        entities = []
        for sentence in doc.sentences:
            for ent in sentence.ents:
                entities.append((ent.text, ent.type))
        
        if entities:
            print(f"  Entities detected: {entities}")
        else:
            print("  No entities detected (this is okay for testing)")
        
        print("\n" + "="*70)
        print("✅ STANZA INSTALLATION COMPLETE!")
        print("="*70)
        print("\nYou can now run the main scripts.")
        print("Example: python main.py --mode batch")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing model: {e}")
        return False

if __name__ == "__main__":
    install_stanza_indonesian()