#!/usr/bin/env python3
"""
Daily PhD Paper Feed
Busca artigos recentes (2025-2026) de mechanistic interpretability,
jailbreaking e prompt injection via arXiv API.
Gera um markdown com 2 abstracts detalhados + 10 titulos.
Salva em daily_phd_feed/ com data no nome do arquivo.
"""

import requests
import random
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

# Configuracoes
SEARCH_QUERIES = [
    "mechanistic interpretability",
    "jailbreaking LLM",
    "prompt injection",
    "adversarial attack language model",
    "sparse autoencoder",
    "activation patching",
    "representation engineering",
    "circuit tracing",
    "model steering",
    "LLM safety",
    "red teaming",
    "feature visualization",
]

# Categorias arXiv relevantes
ARXIV_CATEGORIES = [
    "cs.AI",      # Artificial Intelligence
    "cs.CL",      # Computation and Language (NLP)
    "cs.LG",      # Machine Learning
    "cs.CR",      # Cryptography and Security
    "cs.CV",      # Computer Vision
    "stat.ML",    # Statistics - Machine Learning
]


def fetch_arxiv_papers(query, max_results=15):
    """Busca artigos do arXiv para uma query especifica."""
    query_encoded = query.replace(" ", "+")
    
    # Monta a query com categorias
    cat_filter = " OR ".join([f"cat:{cat}" for cat in ARXIV_CATEGORIES])
    full_query = f"all:{query_encoded} AND ({cat_filter})"
    
    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query={full_query}"
        f"&start=0&max_results={max_results}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        papers = []
        for entry in root.findall('atom:entry', ns):
            title_elem = entry.find('atom:title', ns)
            summary_elem = entry.find('atom:summary', ns)
            published_elem = entry.find('atom:published', ns)
            link_elem = entry.find('atom:id', ns)
            
            title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else "N/A"
            summary = summary_elem.text.strip() if summary_elem is not None else ""
            published = published_elem.text if published_elem is not None else ""
            link = link_elem.text if link_elem is not None else ""
            
            # Pega autores
            authors = []
            for author in entry.findall('atom:author', ns):
                name_elem = author.find('atom:name', ns)
                if name_elem is not None:
                    authors.append(name_elem.text)
            
            # Verifica ano
            year = int(published[:4]) if published else 0
            
            # Pega categorias
            categories = []
            for cat in entry.findall('atom:category', ns):
                term = cat.get('term', '')
                if term:
                    categories.append(term)
            
            papers.append({
                'title': title,
                'summary': summary,
                'year': year,
                'authors': authors,
                'link': link,
                'published': published,
                'categories': categories,
                'source_query': query,
            })
        
        return papers
    except Exception as e:
        print(f"Erro ao buscar '{query}' no arXiv: {e}", file=sys.stderr)
        return []


def is_relevant_paper(paper):
    """Verifica se o artigo e realmente sobre AI/ML interpretability/safety."""
    title = paper['title'].lower()
    summary = paper['summary'].lower() if paper['summary'] else ""
    
    # Palavras-chave que indicam relevancia
    relevant_keywords = [
        "interpretability", "mechanistic", "jailbreak", "prompt injection",
        "adversarial", "alignment", "safety", "red team", "circuit",
        "activation", "autoencoder", "feature", "representation",
        "steering", "unlearning", "editing", "grokking", "superposition",
        "transformer", "language model", "neural network", "deep learning",
        "machine learning", "LLM", "GPT", "inference", "reasoning",
    ]
    
    for kw in relevant_keywords:
        if kw.lower() in title or kw.lower() in summary:
            return True
    
    # Verifica categorias
    ai_cats = ['cs.AI', 'cs.CL', 'cs.LG', 'cs.CR', 'stat.ML']
    for cat in paper.get('categories', []):
        if cat in ai_cats:
            return True
    
    return False


def select_papers(all_papers, num_abstracts=2, num_titles=10):
    """Seleciona artigos para o feed diario."""
    # Filtra apenas artigos relevantes e de 2025-2026
    valid_papers = [
        p for p in all_papers 
        if p['year'] >= 2025 and is_relevant_paper(p) and p['summary']
    ]
    
    if not valid_papers:
        return [], []
    
    # Remove duplicatas por link
    seen_links = set()
    unique_papers = []
    for p in valid_papers:
        if p['link'] not in seen_links:
            seen_links.add(p['link'])
            unique_papers.append(p)
    
    # Embaralha para variedade diaria
    random.shuffle(unique_papers)
    
    abstracts = unique_papers[:num_abstracts]
    titles = unique_papers[num_abstracts:num_abstracts + num_titles]
    
    return abstracts, titles


def generate_markdown(abstracts, titles, date_str):
    """Gera o conteudo markdown do feed."""
    md = f"""# Daily PhD Feed — {date_str}

> Artigos recentes (2025-2026) de Mechanistic Interpretability, Jailbreaking, Prompt Injection e AI Safety.
> Fonte: arXiv API

---

## Abstracts em Destaque ({len(abstracts)} artigos)

"""
    
    for i, paper in enumerate(abstracts, 1):
        title = paper['title']
        authors = ", ".join(paper['authors'][:3]) if paper['authors'] else "Autores desconhecidos"
        if len(paper['authors']) > 3:
            authors += " et al."
        
        year = paper['year']
        link = paper['link']
        summary = paper['summary']
        categories = ", ".join(paper.get('categories', [])[:3])
        
        md += f"""### {i}. {title}

**Autores:** {authors}  
**Ano:** {year} | **Categorias:** {categories}  
**Link:** [arXiv]({link})

**Abstract:**  
{summary}

---

"""
    
    md += f"""## Lista Rapida ({len(titles)} artigos)

> Titulos para explorar depois. Clique no link para ver detalhes.

"""
    
    for i, paper in enumerate(titles, 1):
        title = paper['title']
        authors = ", ".join(paper['authors'][:2]) if paper['authors'] else "Autores desconhecidos"
        if len(paper['authors']) > 2:
            authors += " et al."
        
        year = paper['year']
        link = paper['link']
        
        md += f"{i}. **{title}** — {authors} ({year}) — [arXiv]({link})\n"
    
    md += """

---

*Gerado automaticamente. Para mais detalhes, visite o link do arXiv.*
"""
    
    return md


def main():
    # Diretorio de saida
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "daily_phd_feed"
    output_dir.mkdir(exist_ok=True)
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_file = output_dir / f"feed_{date_str}.md"
    
    print(f"Buscando artigos para {date_str}...")
    
    # Coleta artigos de todas as queries
    all_papers = []
    for query in SEARCH_QUERIES:
        print(f"  Buscando: {query}...")
        papers = fetch_arxiv_papers(query, max_results=15)
        all_papers.extend(papers)
        print(f"    Encontrados: {len(papers)}")
    
    print(f"\nTotal de artigos coletados: {len(all_papers)}")
    
    # Seleciona artigos para o feed
    abstracts, titles = select_papers(all_papers, num_abstracts=2, num_titles=10)
    
    print(f"Artigos relevantes selecionados: {len(abstracts)}")
    print(f"Artigos em lista rapida: {len(titles)}")
    
    if not abstracts and not titles:
        print("Nenhum artigo relevante encontrado.")
        markdown = f"""# Daily PhD Feed — {date_str}

> Nenhum artigo relevante encontrado hoje. Tente novamente amanha.
"""
    else:
        # Gera o markdown
        markdown = generate_markdown(abstracts, titles, date_str)
    
    # Salva o arquivo
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    print(f"\nFeed salvo em: {output_file}")
    print(f"Resumo: {len(abstracts)} abstracts + {len(titles)} titulos")
    
    # Tambem imprime o conteudo para o cronjob capturar
    print("\n" + "="*60)
    print(markdown)


if __name__ == '__main__':
    main()
