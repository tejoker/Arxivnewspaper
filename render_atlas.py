import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import os
import sys

# Config
INPUT_CSV = "output/atlas_data.csv"
OUTPUT_PNG = "output/math_stat_atlas.png"
MAX_EDGES = 50000

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
    if not isinstance(cat_str, str): return CATEGORY_COLORS['other']
    main_cat = cat_str.split(' ')[0]
    if main_cat.startswith('math'): return CATEGORY_COLORS['math']
    if main_cat.startswith('stat'): return CATEGORY_COLORS['stat']
    if main_cat.startswith('cs'): return CATEGORY_COLORS['cs']
    if main_cat.startswith('hep') or main_cat.startswith('astro') or main_cat.startswith('cond-mat') or main_cat.startswith('physics'): return CATEGORY_COLORS['physics']
    return CATEGORY_COLORS['other']

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        sys.exit(1)

    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} papers.")

    # Create mapping for edges
    # We need id -> (x, y)
    print("Indexing coordinates...")
    # Assume arxiv_id is the key
    # Check for duplicates?
    df = df.drop_duplicates(subset=['arxiv_id'])
    
    # Create a nice lookup dictionary
    # id_to_pos = { row['arxiv_id']: (row['x'], row['y']) }
    # Vectorized might be faster but simple loop is fine for 800k in seconds
    id_to_pos = dict(zip(df['arxiv_id'], zip(df['x'], df['y'])))

    # Edges
    print("Generating Edge List (Sampling)...")
    citation_lines = []
    
    # We can iterate and sample.
    sample_rate = MAX_EDGES / (len(df) * 5) # Heuristic: avg 5 citations?
    if sample_rate > 1: sample_rate = 1
    
    # It's faster to just iterate random rows?
    # Let's iterate all and random sample on the fly
    
    count = 0
    # Use simple iteration for safety
    for idx, row in df.iterrows():
        cited_str = row['cited_ids']
        if pd.isna(cited_str) or not cited_str: continue
        
        cited_list = cited_str.split(';')
        if not cited_list: continue
        
        # Draw a link to a random cited paper (or all if we want, but heavy)
        # Let's pick ONE link per paper to keep it manageable? 
        # Or use random probability
        
        for target_id in cited_list:
            if target_id in id_to_pos:
                if np.random.rand() < 0.05: # 5% chance per edge
                    start = (row['x'], row['y'])
                    end = id_to_pos[target_id]
                    citation_lines.append([start, end])
                    count += 1
                    
            if count >= MAX_EDGES: break
        if count >= MAX_EDGES: break

    print(f"Selected {len(citation_lines)} citation edges.")

    # Plot
    print("Rendering Image (High Res)...")
    fig, ax = plt.subplots(figsize=(24, 18), dpi=150) # 150 DPI is decent balance
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#111111')

    # Draw Edges
    if citation_lines:
        lc = LineCollection(citation_lines, colors=EDGE_COLORS['citation'], linewidths=0.1, alpha=0.3, zorder=0)
        ax.add_collection(lc)

    # Draw Nodes
    print("Scatter plot...")
    # Calculate colors
    colors = [get_category_color(x) for x in df['categories']]
    ax.scatter(df['x'], df['y'], c=colors, s=1, alpha=0.8, edgecolors='none', zorder=1)

    # Labels
    # Use 'cluster_region' column to find centers?
    # We can group by cluster_region and calculate mean X,Y
    print("Placing Labels...")
    if 'cluster_region' in df.columns:
        # Filter out NaN/Empty
        cluster_df = df[df['cluster_region'].notna() & (df['cluster_region'] != "")]
        centers = cluster_df.groupby('cluster_region')[['x', 'y']].mean()
        
        for label, row in centers.iterrows():
            ax.text(row['x'], row['y'], label, fontsize=14, color='white', 
                    ha='center', va='center', zorder=2, fontweight='bold', 
                    bbox=dict(facecolor='#111111', alpha=0.6, edgecolor='none'))

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Math (Blue)', markerfacecolor=CATEGORY_COLORS['math'], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Stat (Red)', markerfacecolor=CATEGORY_COLORS['stat'], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='CS (Purple)', markerfacecolor=CATEGORY_COLORS['cs'], markersize=10),
        Line2D([0], [0], color=EDGE_COLORS['citation'], label='Citation Link', lw=1),
    ]
    leg = ax.legend(handles=legend_elements, loc='upper right', facecolor='#111111', edgecolor='white', fontsize=12)
    plt.setp(leg.get_title(), color='white')
    for text in leg.get_texts(): text.set_color('white')

    ax.axis('off')
    
    print(f"Saving to {OUTPUT_PNG}...")
    plt.savefig(OUTPUT_PNG, format='png', facecolor='#111111', bbox_inches='tight')
    plt.close()
    print("Done.")

if __name__ == "__main__":
    main()
