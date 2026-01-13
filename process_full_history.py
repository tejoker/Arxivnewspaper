import time
import pickle
from sqlalchemy import text
from sentence_transformers import SentenceTransformer
from database import get_session, Paper
from benchmarks import extract_benchmarks

# Batch size for processing
BATCH_SIZE = 1000

def process_batch(session, papers, model):
    texts = [f"{p.title} {p.abstract}" for p in papers]
    
    # 1. Compute Embeddings
    embeddings = model.encode(texts)
    
    # 2. Update Papers
    for i, p in enumerate(papers):
        # Store embedding as pickle blob
        p.embedding = pickle.dumps(embeddings[i])
        
        # 3. Extract Benchmarks
        bench_set = extract_benchmarks(texts[i])
        if bench_set:
            p.benchmarks = ",".join(bench_set)
        else:
            p.benchmarks = ""
            
    session.commit()

def process_history():
    print("Initializing Batch Processor...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    session = get_session()
    
    # Check how many papers need processing
    # count where embedding is null
    total = session.query(Paper).count()
    remaining = session.query(Paper).filter(Paper.embedding == None).count()
    
    print(f"Total Papers: {total}")
    print(f"Papers to Process: {remaining}")
    
    start_time = time.time()
    processed_count = 0
    
    try:
        while True:
            # Fetch a batch of unprocessed papers
            batch = session.query(Paper).filter(Paper.embedding == None).limit(BATCH_SIZE).all()
            
            if not batch:
                print("No more papers to process.")
                break
                
            process_batch(session, batch, model)
            
            processed_count += len(batch)
            elapsed = time.time() - start_time
            rate = processed_count / elapsed
            eta = (remaining - processed_count) / rate / 3600 if rate > 0 else 0
            
            print(f"Processed {processed_count} / {remaining}. Rate: {rate:.1f} p/sec. ETA: {eta:.2f} hrs.")
            
    except KeyboardInterrupt:
        print("\nStopping processor...")
    finally:
        session.close()

if __name__ == "__main__":
    process_history()
