import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from database import get_session, Paper

# Constants
OUTPUT_FILE = "output/global_atlas.png"
# 830k papers caused OOM. 100k is sufficient for a dense map and fits in RAM.
SAMPLE_LIMIT = 100000 
BATCH_SIZE = 1000

CATEGORY_COLORS = {
    'math': '#3498db',    # Blue
    'stat': '#e74c3c',    # Red
    'cs': '#9b59b6',      # Purple
    'physics': '#2ecc71', # Green
    'other': '#95a5a6'    # Grey
}

def get_category_color(cat_str):
    if not cat_str: return CATEGORY_COLORS['other']
    main_cat = cat_str.split(' ')[0]
    if main_cat.startswith('math'): return CATEGORY_COLORS['math']
    if main_cat.startswith('stat'): return CATEGORY_COLORS['stat']
    if main_cat.startswith('cs'): return CATEGORY_COLORS['cs']
    if main_cat.startswith('hep') or main_cat.startswith('astro') or main_cat.startswith('cond-mat') or main_cat.startswith('physics'): return CATEGORY_COLORS['physics']
    return CATEGORY_COLORS['other']

def generate_global_map():
    print("--- Journarixv Global Atlas Generator (Memory Optimized) ---")
    session = get_session()
    
    # 1. Count Total
    total_embeddings = session.query(Paper.id).filter(Paper.embedding != None).count()
    print(f"Total available embeddings: {total_embeddings}")
    
    limit = min(total_embeddings, SAMPLE_LIMIT) if SAMPLE_LIMIT else total_embeddings
    print(f"Target sample size: {limit}")
    
    # 2. Pre-allocate Numpy Array (assuming 384 dimensions for all-MiniLM-L6-v2)
    # 100k * 384 * 4 bytes ~= 150MB (Very safe)
    # 830k * 384 * 4 bytes ~= 1.2GB (Safe, but UMAP needs 10x this)
    EMBEDDING_DIM = 384
    X = np.zeros((limit, EMBEDDING_DIM), dtype=np.float32)
    colors = []
    
    # 3. Stream Data
    print("Streaming data from database...")
    query = session.query(Paper.categories, Paper.embedding).filter(Paper.embedding != None)
    
    # Random sampling is hard in SQL efficiently, so we just take the latest (descending ID usually)
    # or just the first N found. For a "Atlas", taking the most recent 100k is actually quite good.
    query = query.limit(limit)
    
    count = 0
    for paper in query.yield_per(BATCH_SIZE):
        try:
            emb = pickle.loads(paper.embedding)
            if len(emb) != EMBEDDING_DIM:
                continue
                
            X[count] = emb
            colors.append(get_category_color(paper.categories))
            count += 1
            
            if count % 10000 == 0:
                print(f"Loaded {count}/{limit}...")
                
            if count >= limit:
                break
        except Exception:
            continue
            
    session.close()
    
    # Resize X if we skipped some errors
    X = X[:count]
    print(f"Final Data Shape: {X.shape}")
    
    # 3. Running UMAP
    print("Running UMAP dimensionality reduction...")
    # low_memory=True is slower but safer
    reducer = umap.UMAP(
        n_neighbors=15, 
        min_dist=0.1, 
        metric='cosine', 
        random_state=42,
        low_memory=True, 
        verbose=True
    )
    embedding_2d = reducer.fit_transform(X)
    
    # 4. Plotting
    print("Generating High-Res Plot...")
    plt.figure(figsize=(20, 15), dpi=300, facecolor='white')
    
    plt.scatter(
        embedding_2d[:, 0], 
        embedding_2d[:, 1], 
        c=colors, 
        s=0.5, 
        alpha=0.6,
        edgecolors='none'
    )
    
    plt.title(f"The Map of Mathematics (Journarixv Atlas - {count} Papers)", fontsize=24)
    plt.axis('off')
    
    # Simple Manual Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Math', markerfacecolor=CATEGORY_COLORS['math'], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Stats', markerfacecolor=CATEGORY_COLORS['stat'], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='CS', markerfacecolor=CATEGORY_COLORS['cs'], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Physics', markerfacecolor=CATEGORY_COLORS['physics'], markersize=10),
    ]
    plt.legend(handles=legend_elements, loc='upper right', title="Fields", fontsize=14)
    
    # 5. Save
    if not os.path.exists("output"):
        os.makedirs("output")
        
    print(f"Saving to {OUTPUT_FILE}...")
    plt.savefig(OUTPUT_FILE, bbox_inches='tight')
    plt.close()
    
    print("Done! Global Atlas generated.")

if __name__ == "__main__":
    generate_global_map()
