from core.engine import SemanticAudioEngine
from core.indexer import UsearchLedger
from gui.interface import ApplicationUI

def main():
    print("Initializing Semantic Audio Engine. Allocating FP16 Tensors to RTX 2050...")
    engine = SemanticAudioEngine()
    
    print("Booting Cryptographic Ledger & Usearch State...")
    indexer = UsearchLedger(data_dir="data")
    
    print("Executing Delta-Update Protocol. Scanning target directory...")
    indexer.ingest_files(engine)
    
    print("Index synchronized. Handing execution context to DearPyGui...")
    ui = ApplicationUI(engine, indexer)
    ui.run()

if __name__ == "__main__":
    main()