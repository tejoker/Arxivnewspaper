import os
import datetime
from jinja2 import Environment, FileSystemLoader

OUTPUT_DIR = "output"

def publish_newspaper(papers, sections, cat_stats, total_papers, section_labels, benchmarks, filiations, graph_data):
    """
    Generates the HTML file for the newspaper.
    papers: List of all Paper objects for the day
    sections: dict {cluster_id: [Paper objects]}
    cat_stats: list of tuples (category, count)
    total_papers: int
    section_labels: dict {cluster_id: "Label Name"}
    benchmarks: dict {benchmark_name: [Paper objects]}
    filiations: dict {ancestor_name: [Paper objects]}
    graph_data: dict {nodes, edges}
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('templates/newspaper.html')
    
    # Use date from first paper, or today if empty
    date_obj = papers[0].published_date if papers else datetime.date.today()

    html_content = template.render(
        date=date_obj.strftime("%A, %B %d, %Y"),
        sections=sections,
        section_labels=section_labels,
        benchmarks=benchmarks,
        filiations=filiations,
        total_papers=total_papers,
        cat_stats=cat_stats,
        graph_data=graph_data
    )
    
    filename = f"newspaper_{date_obj.strftime('%Y-%m-%d')}.html"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    with open(filepath, 'w') as f:
        f.write(html_content)
        
    print(f"Newspaper generated at: {filepath}")
    return filepath

if __name__ == "__main__":
    # Test dummy
    from dataclasses import dataclass
    
    @dataclass
    class DummyPaper:
        title: str
        abstract: str
        authors: str
        link: str
        categories: str
        
    dummy_sections = {
        0: [
            DummyPaper("Attention Is All You Need", "Transformer models...", "Vaswani et al.", "#", "cs.CL"),
            DummyPaper("BERT: Pre-training", "Bidirectional encoder...", "Devlin et al.", "#", "cs.CL"),
        ],
        1: [
            DummyPaper("ResNet: Deep Residual Learning", "Deep layers...", "He et al.", "#", "cs.CV"),
        ]
    }
    
    publish_newspaper(datetime.date.today(), dummy_sections)
