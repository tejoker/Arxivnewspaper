from collections import defaultdict
import json
from database import get_session, Citation, Paper

# Category Colors (approximate mapping for visual distinction)
CATEGORY_COLORS = {
    'math': '#3498db',    # Blue
    'stat': '#e74c3c',    # Red
    'cs': '#9b59b6',      # Purple
    'physics': '#2ecc71', # Green
    'other': '#95a5a6'    # Grey
}

def get_category_color(categories_str):
    if not categories_str: return CATEGORY_COLORS['other']
    main_cat = categories_str.split(' ')[0]
    if main_cat.startswith('math'): return CATEGORY_COLORS['math']
    if main_cat.startswith('stat'): return CATEGORY_COLORS['stat']
    if main_cat.startswith('cs'): return CATEGORY_COLORS['cs']
    if main_cat.startswith('hep') or main_cat.startswith('astro') or main_cat.startswith('cond-mat'): return CATEGORY_COLORS['physics']
    return CATEGORY_COLORS['other']

def build_graph_data(papers, filiations, benchmarks):
    """
    Constructs the Nodes and Edges for the visualization.
    papers: List of Paper objects for the day.
    filiations: Dict { 'Ancestor Title': [Paper_Obj, Paper_Obj] }
    benchmarks: Dict { 'Benchmark Name': [Paper_Obj, Paper_Obj] }
    
    Returns: { 'nodes': [...], 'edges': [...] }
    """
    nodes = []
    edges = []
    
    # Track existing nodes to avoid duplicates
    existing_nodes = set()
    
    # 1. Add Daily Papers as Nodes
    for p in papers:
        if p.id not in existing_nodes:
            nodes.append({
                'id': p.id,
                'label': p.id, # Keep label short (ID), show title in tooltip
                'title': p.title, # Tooltip
                'group': 'paper',
                'color': get_category_color(p.categories)
            })
            existing_nodes.add(p.id)
            
    # 2. Add Benchmark Hubs
    for b_name, p_list in benchmarks.items():
        node_id = f"bench_{b_name}"
        if node_id not in existing_nodes:
            nodes.append({
                'id': node_id,
                'label': b_name,
                'group': 'benchmark',
                'shape': 'star',
                'color': '#f1c40f', # Gold
                'size': 20 + (len(p_list) * 2) # Size depends on popularity
            })
            existing_nodes.add(node_id)
            
        # Add edges Paper <-> Benchmark
        for p in p_list:
            edges.append({
                'from': p.id,
                'to': node_id,
                'color': '#f39c12',
                'dashes': True
            })

    # 3. Add Filiation Ancestors
    # filiations is { 'Ancestor Title': [Paper1, Paper2] }
    # We need a stable ID for ancestors. 
    # Since filiation.py usually uses Titles as keys now (for readability), we'll hash/clean it or looking for ID if possible.
    # Ideally filiations should pass back the ID too, but let's use the Title as ID for the viz node uniqueness.
    
    for anc_title, p_list in filiations.items():
        # Title might be "Specific Calculations (Arxiv 0704.0001)"
        # Let's use the full title string as ID for simplicity
        anc_id = f"anc_{anc_title}"
        
        if anc_id not in existing_nodes:
            nodes.append({
                'id': anc_id,
                'label': anc_title[:20] + "...", # Truncate label
                'title': anc_title,
                'group': 'ancestor',
                'shape': 'dot',
                'color': '#95a5a6', # Grey
                'size': 10 + len(p_list)
            })
            existing_nodes.add(anc_id)
            
        # Add edges Paper -> Ancestor
        for p in p_list:
            edges.append({
                'from': p.id,
                'to': anc_id,
                'arrows': 'to',
                'color': '#7f8c8d'
            })
            
    return {'nodes': nodes, 'edges': edges}
