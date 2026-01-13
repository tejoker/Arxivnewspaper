import requests
import urllib.parse
import time
from collections import defaultdict
from database import get_session, Citation, Paper as PaperModel

# Use polite pool
EMAIL = "journarixv_filiation@example.com"

def get_local_references(paper_id, session):
    """
    Returns a list of Cited Arxiv IDs from local DB.
    """
    results = session.query(Citation.cited_id).filter(Citation.citing_id == paper_id).all()
    # results is list of tuples [('id1',), ('id2',)]
    return [r[0] for r in results]

def get_openalex_references(paper_title):
    """
    Returns a list of OpenAlex IDs that this paper cites.
    """
    safe_title = urllib.parse.quote(paper_title)
    url = f"https://api.openalex.org/works?search={safe_title}&mailto={EMAIL}"
    
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            results = data.get('results', [])
            if results:
                # Assume first match is correct
                return results[0].get('referenced_works', [])
    except:
        pass
    return []

def resolve_local_titles(arxiv_ids, session):
    """
    Given a list of Arxiv IDs, return {id: title} from local DB.
    """
    titles = {}
    papers = session.query(PaperModel).filter(PaperModel.id.in_(arxiv_ids)).all()
    for p in papers:
        titles[p.id] = p.title
    return titles

def resolve_openalex_titles(openalex_ids):
    """
    Given a list of OpenAlex IDs, return {id: title} via API.
    """
    titles = {}
    # Batch query max 50
    for i in range(0, len(openalex_ids), 50):
        batch = openalex_ids[i:i+50]
        ids_str = "|".join(batch)
        url = f"https://api.openalex.org/works?filter=openalex_id:{ids_str}&mailto={EMAIL}&select=id,display_name"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                for work in data.get('results', []):
                    titles[work['id']] = work.get('display_name')
        except:
            pass
    return titles

def find_filiation_overlaps(papers):
    """
    Finds papers that share common ANCESTORS (cited papers).
    Hybrid: Checks local DB first (fast), then OpenAlex (slow).
    """
    ancestor_map = defaultdict(list)
    session = get_session()
    
    print(f"Tracing filiation for {len(papers)} papers...")
    
    # Track which IDs are local (Arxiv ID) and which are OpenAlex
    local_ancestors = set()
    openalex_ancestors = set()
    
    count = 0
    for p in papers:
        # 1. Try Local DB
        refs = get_local_references(p.id, session)
        
        if refs:
            # Found locally!
            for ancestor_id in refs:
                ancestor_map[ancestor_id].append(p)
                local_ancestors.add(ancestor_id)
        else:
            # 2. Fallback to OpenAlex
            # Rate limit handling
            if count % 5 == 0: time.sleep(0.5)
            
            refs = get_openalex_references(p.title)
            if refs:
                for ancestor_id in refs:
                    ancestor_map[ancestor_id].append(p)
                    openalex_ancestors.add(ancestor_id)
                    
        count += 1
        if count % 5 == 0:
            print(f"\rProcessed {count}/{len(papers)}", end="")
            
    print("\nProcessing complete.")
    
    # Filter for significant overlaps
    significant_ancestors = {k: v for k, v in ancestor_map.items() if len(v) > 1}
    
    final_filiation = {}
    
    if significant_ancestors:
        # Resolve Titles
        # Split keys into local vs OpenAlex
        sig_ids = set(significant_ancestors.keys())
        sig_local = list(sig_ids.intersection(local_ancestors))
        sig_openalex = list(sig_ids.intersection(openalex_ancestors))
        
        # Batch Resolve Local
        local_titles = resolve_local_titles(sig_local, session)
        for aid, title in local_titles.items():
            final_filiation[f"{title} (Arxiv {aid})"] = significant_ancestors[aid]
            
        # Batch Resolve OpenAlex
        oa_titles = resolve_openalex_titles(sig_openalex)
        for oid, title in oa_titles.items():
            final_filiation[title] = significant_ancestors[oid]
            
    session.close()
    return final_filiation
