import gzip
import json
import sys
import time
from sqlalchemy import text
from database import get_session, Citation

BATCH_SIZE = 10000

def import_citations(filepath):
    session = get_session()
    
    # Speed optimization
    session.execute(text("PRAGMA journal_mode=WAL;"))
    session.execute(text("PRAGMA synchronous=NORMAL;"))
    
    print(f"Importing citations from {filepath}...")
    
    start_time = time.time()
    count = 0
    imported_edges = 0
    batch = []
    
    try:
        # Open gzip file
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            # The file is a JSON object: {"ID": ["REF1", "REF2"], ...}
            # This is tricky because it's one huge object, not line-by-line JSONL.
            # Loading 32MB JSON into RAM is actually fine (it's not 3GB).
            # Let's try standard json.load.
            
            print("Loading JSON into memory (it's ~32MB compressed, maybe 100-200MB raw)...")
            data = json.load(f)
            print(f"Loaded {len(data)} papers with citations.")
            
            for citing_id, refs in data.items():
                count += 1
                for cited_id in refs:
                    c = Citation(
                        citing_id=citing_id,
                        cited_id=cited_id,
                        source="matt_bierbaum_2019"
                    )
                    batch.append(c)
                    
                    if len(batch) >= BATCH_SIZE:
                        try:
                            session.bulk_save_objects(batch)
                            session.commit()
                            imported_edges += len(batch)
                            batch = []
                            print(f"\rImported {imported_edges} citations...", end="")
                        except Exception as e:
                            session.rollback()
                            print(f"Error skipping batch: {e}")
                            batch = []
                            
            # Final batch
            if batch:
                session.bulk_save_objects(batch)
                session.commit()
                imported_edges += len(batch)

    except Exception as e:
        print(f"\nError reading file: {e}")
        
    duration = time.time() - start_time
    print(f"\n--- Import Complete ---")
    print(f"Papers processed: {count}")
    print(f"Edges (citations): {imported_edges}")
    print(f"Time: {duration:.2f}s")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python import_citations.py <path_to_internal-references.json.gz>")
    else:
        import_citations(sys.argv[1])
