import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA
from collections import Counter, defaultdict
from database import get_session, Paper, Citation
import gc
import csv

# Configuration
OUTPUT_IMAGE = "output/math_stat_atlas.svg"
OUTPUT_DATA = "output/atlas_data.csv"
TEMP_METADATA = "output/temp_metadata.csv"
TEMP_VECTORS = "output/temp_vectors.npy"
SAMPLE_LIMIT = None 
BATCH_SIZE = 10000 
PCA_COMPONENTS = 50 
MAX_EDGES_TO_PLOT = 50000 

CATEGORY_COLORS = {
    'math': '#3498db',    # Blue
    'stat': '#e74c3c',    # Red
    'cs': '#9b59b6',      # Purple
    'physics': '#2ecc71', # Green
    'other': '#95a5a6'    # Grey
}

EDGE_COLORS = {
    'citation': '#bdc3c7', 
    'benchmark': '#f1c40f' 
}

def get_category_color(cat_str):
    if not cat_str: return CATEGORY_COLORS['other']
    main_cat = cat_str.split(' ')[0]
    if main_cat.startswith('math'): return CATEGORY_COLORS['math']
    if main_cat.startswith('stat'): return CATEGORY_COLORS['stat']
    if main_cat.startswith('cs'): return CATEGORY_COLORS['cs']
    if main_cat.startswith('hep') or main_cat.startswith('astro') or main_cat.startswith('cond-mat') or main_cat.startswith('physics'): return CATEGORY_COLORS['physics']
    return CATEGORY_COLORS['other']

def get_primary_category(cat_str):
    if not cat_str: return "Uncategorized"
    return cat_str.split(' ')[0]

def build_atlas():
    print("--- Journarixv Atlas Builder (Disk-Backed / Ultra-Low RAM) ---")
    if not os.path.exists("output"): os.makedirs("output")
    
    session = get_session()
    
    # 1. Incremental PCA Training
    print(f"Phase 1: Training PCA (Target: {PCA_COMPONENTS} dims)...")
    ipca = IncrementalPCA(n_components=PCA_COMPONENTS, batch_size=BATCH_SIZE)
    query = session.query(Paper.categories, Paper.embedding).filter(Paper.embedding != None)
    
    batch_buffer = []
    count = 0
    # Train on first 100k or all? Training on all is slow but ensures global fit.
    # Given tight memory, let's train on 50k and assume distribution holds.
    TRAIN_LIMIT = 50000
    
    for p in query.yield_per(BATCH_SIZE):
        cats = p.categories or ""
        if not any(c.startswith('math') or c.startswith('stat') for c in cats.split()): continue
        try:
            emb = pickle.loads(p.embedding)
            if len(emb) != 384: continue
            batch_buffer.append(emb)
        except: continue
        
        if len(batch_buffer) >= BATCH_SIZE:
            ipca.partial_fit(batch_buffer)
            count += len(batch_buffer)
            batch_buffer = []
            sys.stdout.write(f"\rPCA Train: {count}")
            # Train only on a subset to save time/resources if we assume homogeneity
            if count >= TRAIN_LIMIT: break
            
    if batch_buffer: ipca.partial_fit(batch_buffer)
    del batch_buffer
    gc.collect()
    print("\nPCA Trained.")

    # 2. Transform & Stream to Disk
    print("Phase 2: Streaming Metadata to Disk & Reducing Vectors...")
    query = session.query(Paper.id, Paper.title, Paper.categories, Paper.embedding, Paper.benchmarks).filter(Paper.embedding != None)
    
    # We write metadata to CSV instantly. No lists in RAM.
    meta_file = open(TEMP_METADATA, 'w', newline='', encoding='utf-8')
    writer = csv.writer(meta_file)
    writer.writerow(['idx', 'id', 'title', 'categories', 'benchmarks', 'color']) # Header
    
    # We collect reduced vectors in a list (ints/floats are cheaper than strings, but still...)
    # 830k * 50 * 4 bytes = 160MB in pure C-array. Python list overhead?
    # Better to append to numpy array or list of arrays.
    
    vectors_list = []
    batch_buffer = [] # Raw embeddings (384)
    meta_buffer = [] 
    
    total_idx = 0
    id_to_idx = {} # We DO need this for edges. 830k ints key -> int val. ~830k * 100 bytes ~ 80MB. Acceptable.
    
    for p in query.yield_per(BATCH_SIZE):
        cats = p.categories or ""
        if not any(c.startswith('math') or c.startswith('stat') for c in cats.split()): continue
        try:
            emb = pickle.loads(p.embedding)
            if len(emb) != 384: continue
            
            batch_buffer.append(emb)
            meta_buffer.append((p.id, p.title, cats, p.benchmarks))
        except: continue
        
        if len(batch_buffer) >= BATCH_SIZE:
            # Transform
            reduced = ipca.transform(batch_buffer)
            # Write Vectors to list (keep in RAM, it's small enough now)
            vectors_list.append(reduced)
            
            # Write Metadata
            for i, vec in enumerate(reduced):
                pid, tit, cat, bench = meta_buffer[i]
                writer.writerow([total_idx, pid, tit, cat, bench, get_category_color(cat)])
                id_to_idx[pid] = total_idx
                total_idx += 1
            
            batch_buffer = []
            meta_buffer = []
            sys.stdout.write(f"\rProcessed: {total_idx}")
            
            if SAMPLE_LIMIT and total_idx >= SAMPLE_LIMIT: break

    # Flush
    if batch_buffer:
        reduced = ipca.transform(batch_buffer)
        vectors_list.append(reduced)
        for i, vec in enumerate(reduced):
            pid, tit, cat, bench = meta_buffer[i]
            writer.writerow([total_idx, pid, tit, cat, bench, get_category_color(cat)])
            id_to_idx[pid] = total_idx
            total_idx += 1
            
    meta_file.close()
    session.close()
    
    # Join vectors
    if not vectors_list: return
    X = np.vstack(vectors_list)
    del vectors_list
    gc.collect()
    
    print(f"\nDataset Ready: {X.shape}. RAM Cleared.")
    
    # 3. Parametric UMAP (Train on subset, Transform rest)
    print("Phase 3: Parametric UMAP (Embedded System Mode)...")
    
    # Randomly shuffle data indices to ensure training set is representative
    # But wait, our X is sorted by ID usually. Shuffling is needed.
    # We already have X in RAM (166MB).
    
    total_samples = X.shape[0]
    # Memory Calc: 300k samples -> ~1.5GB Peak.
    # User has 5.4GB Free. This is safe and statistically very robust (>1/3 of all data).
    TRAIN_SIZE = 300000 
    
    indices = np.arange(total_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_idx = indices[:TRAIN_SIZE]
    rest_idx = indices[TRAIN_SIZE:]
    
    X_train = X[train_idx]
    
    print(f"Training UMAP on {TRAIN_SIZE} samples...")
    # Standard UMAP fit
    mapper = umap.UMAP(
        n_neighbors=15, 
        min_dist=0.1, 
        metric='cosine', 
        random_state=42, 
        low_memory=True,
        verbose=True
    ).fit(X_train)
    
    print("Projecting remaining samples (Batched Transform)...")
    
    # We need to construct the full 2D embedding array in original order
    # Let's create an empty container
    embedding_2d = np.zeros((total_samples, 2), dtype=np.float32)
    
    # Fill in training (mapped) data
    embedding_2d[train_idx] = mapper.embedding_
    
    # Transform the rest in batches to save RAM during projection
    TRANSFORM_BATCH = 20000
    total_rest = len(rest_idx)
    
    for i in range(0, total_rest, TRANSFORM_BATCH):
        batch_indices = rest_idx[i : i + TRANSFORM_BATCH]
        X_batch = X[batch_indices]
        
        # Transform
        proj = mapper.transform(X_batch)
        embedding_2d[batch_indices] = proj
        
        sys.stdout.write(f"\rProjected {min(i + TRANSFORM_BATCH, total_rest)}/{total_rest}")
        sys.stdout.flush()
        gc.collect()
        
    print("\nUMAP Complete.")
    
    # Clean up source X to free 160MB
    del X
    del mapper
    gc.collect()
    
    # 4. Clustering Labels
    print("Phase 4: Clustering...")
    kmeans = KMeans(n_clusters=20, random_state=42)
    labels = kmeans.fit_predict(embedding_2d)
    centers = kmeans.cluster_centers_
    
    # How to name clusters? We need categories. Read Streaming from CSV? 
    # Or just random access? Reading CSV 800k times is slow.
    # We can do a quick pass of the CSV to build label map.
    print(" Naming Regions (Scanning Metadata)...")
    cluster_cats = defaultdict(list)
    
    # We need to map row_idx -> label -> category
    # Scan CSV once
    with open(TEMP_METADATA, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row['idx'])
            cat = row['categories']
            lbl = labels[idx]
            cluster_cats[lbl].append(cat.split(' ')[0])
            
    cluster_names = {}
    name_map = {
            'math.CO': 'Combinatorics', 'math.AP': 'Analysis (PDE)', 'math.AG': 'Alg. Geometry',
            'math.NT': 'Number Theory', 'math.PR': 'Probability', 'math.DS': 'Dynamical Sys',
            'stat.ML': 'Machine Learning', 'stat.ME': 'Methodology', 'stat.AP': 'Applied Stat',
            'math.GT': 'Topology', 'math.QA': 'Quantum Alg', 'math.RT': 'Representation',
            'math.FA': 'Functional Analysis', 'math.DG': 'Diff. Geometry', 'math.LO': 'Logic',
            'math.OC': 'Optimization', 'math.ST': 'Statistics Th', 'cs.IT': 'Info Theory'
    }
    
    for i in range(20):
        if not cluster_cats[i]:
            cluster_names[i] = ""
        else:
            top = Counter(cluster_cats[i]).most_common(1)[0][0]
            cluster_names[i] = name_map.get(top, top)
            
    del cluster_cats
    gc.collect()
    
    # 5. Edges
    print("Phase 5: Citations...")
    session = get_session()
    citation_query = session.query(Citation.citing_id, Citation.cited_id).yield_per(50000)
    
    # We need edges for plotting AND for export (cited_ids column).
    # Export: We need a map. Adjacency list by string ID or Index? 
    # CSV export needs string ID. Map: idx -> [cited_string_id_1, ...]
    
    # Since RAM is tight:
    # 1. Collect Edges for Plotting (Sampled Limit). Raw list.
    # 2. Collect Edges for CSV Export? Full list (830k * 20 * 8 bytes) might be heavy (150MB). Acceptable.
    
    plot_edges_c = []
    export_citations = defaultdict(list) # idx -> [cited_string_ids]
    
    # Reverse map for export
    idx_to_str_id = {v: k for k, v in id_to_idx.items()}
    
    c_count = 0
    for citing, cited in citation_query:
        if citing in id_to_idx and cited in id_to_idx:
            u, v = id_to_idx[citing], id_to_idx[cited]
            
            # For Plot (Sampled)
            if len(plot_edges_c) < MAX_EDGES_TO_PLOT:
               plot_edges_c.append((u, v))
            elif np.random.rand() < 0.01: # Reservoir sample roughly
               plot_edges_c[np.random.randint(0, MAX_EDGES_TO_PLOT)] = (u, v)
               
            # For Export (idx -> string)
            export_citations[u].append(idx_to_str_id[v])
            c_count += 1
            
    print(f"Citations: {c_count}")
    
    # Benchmarks
    print("Benchmarks...")
    plot_edges_b = []
    # Scan metadata csv again? No, we didn't store raw benchmarks in RAM.
    # Creating edges requires grouping by benchmark.
    # Scan CSV -> Build {bench: [indices]}
    bench_map = defaultdict(list)
    with open(TEMP_METADATA, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
             b_str = row['benchmarks']
             if not b_str: continue
             idx = int(row['idx'])
             try:
                cleaned = b_str.replace('[','').replace(']','').replace("'", "").replace('"','')
                for b in [x.strip() for x in cleaned.split(',') if x.strip()]:
                    bench_map[b].append(idx)
             except: continue
             
    for b_name, indices in bench_map.items():
        if len(indices) < 2: continue
        hub = indices[0]
        for p_idx in indices[1:]:
            if len(plot_edges_b) < MAX_EDGES_TO_PLOT:
                plot_edges_b.append((hub, p_idx))
                
    del bench_map
    gc.collect()

    # 6. Plotting
    print("Phase 6: Plotting SVG...")
    fig, ax = plt.subplots(figsize=(24, 18))
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#111111')
    
    # Edges
    from matplotlib.collections import LineCollection
    if plot_edges_c:
        lines = [[embedding_2d[u], embedding_2d[v]] for u, v in plot_edges_c]
        lc = LineCollection(lines, colors=EDGE_COLORS['citation'], linewidths=0.1, alpha=0.3, zorder=0)
        ax.add_collection(lc)
    if plot_edges_b:
        lines = [[embedding_2d[u], embedding_2d[v]] for u, v in plot_edges_b]
        lc = LineCollection(lines, colors=EDGE_COLORS['benchmark'], linewidths=0.3, alpha=0.6, zorder=0)
        ax.add_collection(lc)
        
    # Nodes (Scan CSV for colors to avoid big list in RAM? No, plot needs array)
    # We need colors aligned with embedding_2d.
    # Scan CSV one last time to get colors list? Or used saved index?
    # We saved `get_category_color(cat)` in CSV.
    colors = []
    with open(TEMP_METADATA, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            colors.append(row['color'])
            
    ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=colors, s=1, alpha=0.8, edgecolors='none', zorder=1)
    
    # Labels
    for i in range(20):
        cx, cy = centers[i]
        ax.text(cx, cy, cluster_names[i], fontsize=14, color='white', ha='center', va='center', zorder=2, fontweight='bold', bbox=dict(facecolor='#111111', alpha=0.6, edgecolor='none'))
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Math (Blue)', markerfacecolor=CATEGORY_COLORS['math'], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Stat (Red)', markerfacecolor=CATEGORY_COLORS['stat'], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='CS (Purple)', markerfacecolor=CATEGORY_COLORS['cs'], markersize=10),
        Line2D([0], [0], color=EDGE_COLORS['citation'], label='Citation Link', lw=1),
        Line2D([0], [0], color=EDGE_COLORS['benchmark'], label='Shared Benchmark', lw=1),
    ]
    leg = ax.legend(handles=legend_elements, loc='upper right', facecolor='#111111', edgecolor='white', fontsize=12)
    plt.setp(leg.get_title(), color='white')
    for text in leg.get_texts(): text.set_color('white')
    
    ax.axis('off')
    plt.savefig(OUTPUT_IMAGE, format='svg', facecolor='#111111', bbox_inches='tight')
    plt.close()
    
    # 7. Final Export Merge
    print("Phase 7: Exporting Final CSV...")
    # Join Metadata CSV + Embedding + Computed Citations
    with open(TEMP_METADATA, 'r', encoding='utf-8') as f_in, open(OUTPUT_DATA, 'w', newline='', encoding='utf-8') as f_out:
        reader = csv.DictReader(f_in)
        # Headers
        fieldnames = ['arxiv_id', 'title', 'categories', 'x', 'y', 'benchmarks', 'cluster_region', 'cited_ids']
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, row in enumerate(reader):
            # idx = int(row['idx']) # Matches i
            writer.writerow({
                'arxiv_id': row['id'],
                'title': row['title'],
                'categories': row['categories'],
                'x': embedding_2d[i, 0],
                'y': embedding_2d[i, 1],
                'benchmarks': row['benchmarks'],
                'cluster_region': cluster_names[labels[i]],
                'cited_ids': ";".join(export_citations[i]) # List of strings
            })
            
    # Cleanup
    if os.path.exists(TEMP_METADATA): os.remove(TEMP_METADATA)
    print("Done! Ultra-Optimized Run Complete.")

if __name__ == "__main__":
    build_atlas()
