import argparse
import datetime
import sys
from datetime import timedelta

# Import modules to ensure database is init
from database import init_db, get_session, Paper
from fetcher import fetch_papers_by_date, save_papers
from analyzer import cluster_papers
from publisher import publish_newspaper

def main():
    parser = argparse.ArgumentParser(description='Journarixv: Your Local Arxiv Newspaper')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD). Default: yesterday.')
    parser.add_argument('--category', type=str, default="cs.*", help='Arxiv category (default: cs.*).')
    parser.add_argument('--force-fetch', action='store_true', help='Force fetching from Arxiv even if papers exist in DB.')
    
    args = parser.parse_args()
    
    # 0. Setup
    init_db()
    
    # Determine Date
    if args.date:
        try:
            target_date = datetime.datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print("Error: Invalid date format. Use YYYY-MM-DD.")
            sys.exit(1)
    else:
        target_date = datetime.date.today() - timedelta(days=1)
        
    print(f"=== Journarixv Daily Edition: {target_date} ===")
    
    # 1. Fetching (Check DB first)
    session = get_session()
    
    # Simple check: do we have papers for this date?
    # Note: Arxiv date can be 'published' or 'updated'. We stored 'published_date'.
    # A more robust check might be complex, but let's see count.
    existing_papers = session.query(Paper).filter(Paper.published_date == target_date).all()
    
    if args.force_fetch or not existing_papers:
        print(f"Fetching papers from Arxiv for {target_date}...")
        fetched_papers = fetch_papers_by_date(target_date, category=args.category)
        save_papers(fetched_papers)
        # Re-query to get objects attached to session/with IDs if needed (though save_papers handles it)
        # For simplicity, we can just use fetched_papers, but better to query DB to be consistent
        existing_papers = session.query(Paper).filter(Paper.published_date == target_date).all()
    else:
        print(f"Found {len(existing_papers)} existing papers in local DB. Skipping fetch (use --force-fetch to override).")
        
    if not existing_papers:
        print("No papers found for this date. Exiting.")
        return

    # 2. Intelligence (Clustering)
    print("Analyzing and clustering papers...")
    # This might take time if generating embeddings for the first time
    clusters = cluster_papers(existing_papers)
    
    # Generate labels for clusters
    from analyzer import extract_cluster_keywords
    cluster_labels = {}
    print("Generating topic labels...")
    for cid, p_list in clusters.items():
        label = extract_cluster_keywords(p_list)
        cluster_labels[cid] = label
        print(f"  Cluster {cid}: {label}")

    # 2.5 Benchmark Analysis
    from benchmarks import find_benchmark_overlaps
    print("Finding Benchmark Competitors...")
    common_benchmarks = find_benchmark_overlaps(existing_papers)
    print(f"Found {len(common_benchmarks)} shared benchmarks.")
    
    
    # 2.6 Filiation Analysis (Hybrid)
    from filiation import find_filiation_overlaps
    print("Tracing Citation/Filiation Graph...")
    # Use ALL papers now that we have offline DB
    filiations = find_filiation_overlaps(existing_papers)
    print(f"Found {len(filiations)} common ancestors.")
    
    # 2.7 Graph Visualization Construction
    from graph_builder import build_graph_data
    print("Building Network Graph...")
    graph_data = build_graph_data(existing_papers, filiations, common_benchmarks)

    # 3. Publishing
    print("Generating newspaper...")
    # Calculate stats
    total_papers = len(existing_papers)
    from collections import Counter
    cat_stats = Counter([p.categories.split(' ')[0] for p in existing_papers]).most_common(5)
    
    filepath = publish_newspaper(
        existing_papers, 
        clusters, 
        cat_stats, 
        total_papers, 
        cluster_labels, 
        common_benchmarks, 
        filiations, 
        graph_data
    )
    
    print(f"\nDone! Open your newspaper here:\nfile://{os.path.abspath(filepath)}")

if __name__ == "__main__":
    import os
    main()
