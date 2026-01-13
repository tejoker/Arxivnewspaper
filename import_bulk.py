import json
import os
import sys
import time
from datetime import datetime
from sqlalchemy.exc import IntegrityError
from sqlalchemy import text
from database import get_session, Paper

# Batch size for database inserts
BATCH_SIZE = 5000

def parse_date(date_str):
    try:
        if not date_str:
            return None
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None

def import_bulk_file(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return

    session = get_session()
    
    # Enable WAL mode for faster writes
    session.execute(text("PRAGMA journal_mode=WAL;"))
    session.execute(text("PRAGMA synchronous=NORMAL;"))
    
    print(f"Starting bulk import from {filepath}...")
    print(f"Filtering for categories: math.*, stat.*")
    
    start_time = time.time()
    count = 0
    imported = 0
    batch = []
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                count += 1
                if count % 10000 == 0:
                    print(f"Scanned {count} papers... (Imported {imported})")
                
                categories = doc.get('categories', '')
                
                # Filter Logic
                is_target = False
                cat_list = categories.split(' ')
                for cat in cat_list:
                    if cat.startswith('math') or cat.startswith('stat'):
                        is_target = True
                        break
                
                if not is_target:
                    continue
                
                # Handle authors
                authors = doc.get('authors_parsed', [])
                if authors:
                    author_str = ", ".join([f"{a[1]} {a[0]}".strip() for a in authors])
                else:
                    author_str = doc.get('authors', '')

                pid = doc.get('id')
                link = f"http://arxiv.org/abs/{pid}"
                
                # Published Date
                versions = doc.get('versions', [])
                pub_date = None
                if versions:
                    try:
                        v1_date = versions[0].get('created')
                        dt = datetime.strptime(v1_date, "%a, %d %b %Y %H:%M:%S %Z")
                        pub_date = dt.date()
                    except:
                        pass
                
                if not pub_date:
                    pub_date = parse_date(doc.get('update_date'))

                paper = Paper(
                    id=pid,
                    title=doc.get('title', '').replace('\n', ' ').strip(),
                    abstract=doc.get('abstract', '').replace('\n', ' ').strip(),
                    authors=author_str[:500],
                    link=link,
                    published_date=pub_date,
                    updated_date=parse_date(doc.get('update_date')),
                    categories=categories
                )
                
                batch.append(paper)
                
                if len(batch) >= BATCH_SIZE:
                    try:
                        session.bulk_save_objects(batch)
                        session.commit()
                        imported += len(batch)
                        batch = []
                    except IntegrityError:
                        session.rollback()
                        for p in batch:
                            try:
                                session.merge(p)
                            except:
                                pass
                        session.commit()
                        imported += len(batch)
                        batch = []
                        
        # Final batch
        if batch:
            session.bulk_save_objects(batch)
            session.commit()
            imported += len(batch)

    except KeyboardInterrupt:
        print("\nStopping import...")
        
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n--- Import Complete ---")
    print(f"Scanned Total: {count}")
    print(f"Imported (Math/Stat): {imported}")
    print(f"Time Taken: {duration:.2f}s")
    if duration > 0:
        print(f"Rate: {count/duration:.1f} papers/sec")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python import_bulk.py <path_to_arxiv_snapshot.json>")
        print("Example: python import_bulk.py arxiv-metadata-oai-snapshot.json")
    else:
        import_bulk_file(sys.argv[1])
