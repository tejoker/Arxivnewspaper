import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import datetime
from datetime import timedelta
import time
from database import get_session, Paper, init_db

# Arxiv API Base URL
ARXIV_API_URL = 'http://export.arxiv.org/api/query'

NAMESPACES = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}

def fetch_papers_by_date(date_obj, category="cs.*", page_size=500):
    """
    Fetches papers submitted on a specific date.
    Note: Arxiv API 'submittedDate' is precise. We search for range [date, date+1).
    """
    
    # Construct query
    # We want papers SUBMITTED on this day.
    # formatting: YYYYMMDDHHMM
    start_date = date_obj.strftime("%Y%m%d0000")
    end_date = (date_obj + timedelta(days=1)).strftime("%Y%m%d0000")
    
    # Query: cat:category AND submittedDate:[start TO end]
    query = f'cat:{category} AND lastUpdatedDate:[{start_date} TO {end_date}]'
    
    all_papers = []
    start = 0
    
    print(f"Starting fetch for {date_obj}...")
    
    while True:
        params = {
            'search_query': query,
            'start': start,
            'max_results': page_size,
            'sortBy': 'lastUpdatedDate',
            'sortOrder': 'descending'
        }
        
        data = urllib.parse.urlencode(params)
        url = f"{ARXIV_API_URL}?{data}"
        
        print(f"Fetching batch start={start}...")
        
        try:
            with urllib.request.urlopen(url) as response:
                xml_data = response.read()
                batch = parse_arxiv_response(xml_data)
                
                if not batch:
                    print("No more papers found.")
                    break
                
                all_papers.extend(batch)
                start += len(batch)
                
                # Check if we got fewer than requested, meaning we reached the end
                if len(batch) < page_size:
                    break
                    
                time.sleep(3) # Be nice to the API
                
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    return all_papers

def parse_arxiv_response(xml_data):
    root = ET.fromstring(xml_data)
    papers = []
    
    for entry in root.findall('atom:entry', NAMESPACES):
        id_url = entry.find('atom:id', NAMESPACES).text
        arxiv_id = id_url.split('/')[-1]
        
        title = entry.find('atom:title', NAMESPACES).text.strip().replace('\n', ' ')
        summary = entry.find('atom:summary', NAMESPACES).text.strip().replace('\n', ' ')
        
        authors = []
        for author in entry.findall('atom:author', NAMESPACES):
            authors.append(author.find('atom:name', NAMESPACES).text)
            
        published = entry.find('atom:published', NAMESPACES).text
        updated = entry.find('atom:updated', NAMESPACES).text
        
        # Parse dates strings to date objects (YYYY-MM-DD)
        published_date = datetime.datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ").date()
        updated_date = datetime.datetime.strptime(updated, "%Y-%m-%dT%H:%M:%SZ").date()
        
        primary_cat = entry.find('arxiv:primary_category', NAMESPACES)
        category = primary_cat.attrib['term'] if primary_cat is not None else "Unknown"

        paper = Paper(
            id=arxiv_id,
            title=title,
            abstract=summary,
            authors=", ".join(authors),
            link=id_url,
            published_date=published_date,
            updated_date=updated_date,
            categories=category
        )
        papers.append(paper)
        
    return papers

def save_papers(papers):
    session = get_session()
    new_count = 0
    for p in papers:
        # Check if exists
        exists = session.query(Paper).filter_by(id=p.id).first()
        if not exists:
            session.add(p)
            new_count += 1
    session.commit()
    print(f"Saved {new_count} new papers out of {len(papers)} fetched.")
    session.close()

if __name__ == "__main__":
    import argparse
    import sys
    
    init_db()
    
    parser = argparse.ArgumentParser(description='Fetch Arxiv papers for a specific date.')
    parser.add_argument('--date', type=str, help='Date in YYYY-MM-DD format (default: yesterday)')
    parser.add_argument('--category', type=str, default="cs.*", help='Arxiv category (default: cs.*)')
    
    args = parser.parse_args()
    
    if args.date:
        try:
            target_date = datetime.datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print("Error: Invalid date format. Use YYYY-MM-DD.")
            sys.exit(1)
    else:
        target_date = datetime.date.today() - timedelta(days=1)
    
    print(f"--- Running Fetcher for {target_date} (Category: {args.category}) ---")
    papers = fetch_papers_by_date(target_date, category=args.category)
    print(f"Total Found: {len(papers)} papers.")
    save_papers(papers)
