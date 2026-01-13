import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from database import get_session, Paper
import json

# Load model locally (will download on first run)
# 'all-MiniLM-L6-v2' is small and fast
MODEL_NAME = 'all-MiniLM-L6-v2'
_model = None

def get_model():
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME}...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def generate_embeddings(papers):
    """
    Generates embeddings for a list of papers based on Title + Abstract.
    """
    model = get_model()
    texts = [f"{p.title}\n{p.abstract}" for p in papers]
    
    print(f"Generating embeddings for {len(papers)} papers...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def cluster_papers(papers, distance_threshold=1.5):
    """
    Clusters papers based on their embeddings.
    Returns a dict {cluster_id: [papers]}
    """
    if not papers:
        return {}

    # Extract embeddings
    # Note: Embeddings in DB are blobs (pickled or raw bytes). Assuming we handle that.
    # For this function, we assume 'papers' have their embeddings accessible or passed in.
    # But usually we generate them if missing.
    
    # Check if we have embeddings, if not generate
    papers_to_encode = [p for p in papers if p.embedding is None]
    if papers_to_encode:
        new_embeddings = generate_embeddings(papers_to_encode)
        
        # Save to DB
        session = get_session()
        for p, emb in zip(papers_to_encode, new_embeddings):
            # We need to attach this to the persistent object
            # Re-querying or attaching to session might be needed if objects are detached
            # For simplicity, assuming 'p' are attached or we update them
            p.embedding = pickle.dumps(emb)
            # define p.cluster_id if we want to store it (schema update needed?)
            # For now, we just return clusters dynamically
        
        # If papers were passed from a session query, they should be updated.
        # We might need to commit if we want to save embeddings permanently.
        # session.commit() - Ideally handled by caller or we accept a session.
    
    # Now valid_papers are those with embeddings (deserialize first)
    valid_papers = []
    X = []
    
    for p in papers:
        if p.embedding:
            try:
                emb = pickle.loads(p.embedding)
                X.append(emb)
                valid_papers.append(p)
            except:
                pass
                
    if not X:
        return {}
    
    X = np.array(X)
    X = X / np.linalg.norm(X, axis=1, keepdims=True) # Normalize
    
    # Clustering
    # Distance threshold determines "tightness" of clusters.
    # Cosine distance (1 - similarity). Since normalized, Euclidean distance is related.
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,  # Tune this!
        metric='euclidean',
        linkage='ward'
    )
    labels = clustering.fit_predict(X)
    
    clusters = {}
    for label, paper in zip(labels, valid_papers):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(paper)
        
    return clusters

def extract_cluster_keywords(papers, top_n=3):
    """
    Extracts top keywords/phrases from a list of papers to name the cluster.
    Uses TF-IDF to find distinctive terms.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    
    if not papers:
        return "Unknown Topic"
        
    # Combine title (weighted more) and abstract
    # We construct a single document for this cluster
    cluster_text = " ".join([f"{p.title} {p.title} {p.title} {p.abstract}" for p in papers])
    
    # We need a corpus to compare against to find *distinctive* words (TF-IDF).
    # Ideally, we would fit TF-IDF on ALL papers for the day, then transform this cluster.
    # For simplicity/efficiency without passing the global corpus every time, 
    # we can just use frequency (CountVectorizer) with stop words, 
    # OR we assume the caller passes context. 
    # Let's simple frequency of bigrams/trigrams first, as "Benchmark Names" are often N-grams.
    
    vectorizer = CountVectorizer(
        stop_words='english', 
        ngram_range=(2, 3), # Bigrams and Trigrams (e.g. "large language", "object detection")
        max_features=10
    )
    
    try:
        X = vectorizer.fit_transform([cluster_text])
        # Sum counts
        counts = X.sum(axis=0).A1
        words = vectorizer.get_feature_names_out()
        
        # Sort by count
        freqs = list(zip(words, counts))
        freqs.sort(key=lambda x: x[1], reverse=True)
        
        # Pick top N
        top_terms = [f[0].title() for f in freqs[:top_n]]
        return ", ".join(top_terms)
        
    except ValueError:
        # Happens if empty vocabulary (e.g. all stop words)
        return "General Topic"

if __name__ == "__main__":
    # Test Block
    session = get_session()
    # Fetch some papers
    papers = session.query(Paper).limit(50).all()
    
    if not papers:
        print("No papers in DB to analyze.")
    else:
        clusters = cluster_papers(papers)
        print(f"Found {len(clusters)} clusters.")
        for cid, cluster_papers_list in clusters.items():
            topic_name = extract_cluster_keywords(cluster_papers_list)
            print(f"-- Cluster {cid}: {topic_name} ({len(cluster_papers_list)} papers) --")
            for p in cluster_papers_list[:3]:
                print(f"   - {p.title}")
    session.close()
