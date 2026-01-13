from collections import defaultdict
import re

BENCHMARK_KEYWORDS = {
    "MMLU", "HumanEval", "GSM8K", "ImageNet", "COCO", "CIFAR", "Big-Bench", 
    "TruthfulQA", "HellaSwag", "ARC", "MBPP", "Spider", "SQuAD", "GLUE", "SuperGLUE",
    "AlpacaEval", "MT-Bench", "Chatbot Arena"
}

def extract_benchmarks(text):
    """
    Extracts explicit benchmark mentions from text.
    Returns a set of benchmark nanmes.
    """
    found = set()
    # Check for known benchmarks (case sensitive usually fine for acronyms)
    for bench in BENCHMARK_KEYWORDS:
        if bench in text:
            found.add(bench)
            
    # Heuristic for unknown benchmarks: 
    # Look for patterns like "XXX benchmark", "XXX dataset"
    # match = re.findall(r'([A-Z][a-zA-Z0-9-]+) benchmark', text)
    # found.update(match)
    
    return found

def find_benchmark_overlaps(papers):
    """
    Finds papers that discuss the same benchmarks.
    Returns dict: {benchmark_name: [paper_obj, paper_obj]}
    """
    benchmark_map = defaultdict(list)
    
    print("Scanning papers for benchmarks...")
    for p in papers:
        # Combine title and abstract
        text = f"{p.title} {p.abstract}"
        benchmarks = extract_benchmarks(text)
        
        for b in benchmarks:
            benchmark_map[b].append(p)
            
    # Filter out singletons (benchmarks mentioned by only one paper)
    # They don't create "filiation" or "competition" links
    overlaps = {k: v for k, v in benchmark_map.items() if len(v) > 1}
    
    return overlaps
