#!/usr/bin/env python3
"""
Systematic Literature Review Tool
Ferramenta para conducao de revisoes sistematicas da literatura em AI/ML.
Segue metodologia PRISMA (Preferred Reporting Items for Systematic Reviews and Meta-Analyses).

Uso:
    python systematic_review.py --config review_config.yaml
    python systematic_review.py --init my_review  # cria estrutura de diretorios
    python systematic_review.py --run my_review/review_config.yaml
"""

import argparse
import yaml
import json
import csv
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Set
import time


@dataclass
class ReviewConfig:
    """Configuracao da revisao sistematica."""
    title: str
    research_question: str
    objectives: str
    
    # Criterios de inclusao
    inclusion_criteria: List[str]
    
    # Criterios de exclusao
    exclusion_criteria: List[str]
    
    # Strings de busca
    search_queries: List[str]
    
    # Fontes de dados
    sources: List[str] = field(default_factory=lambda: ["arxiv", "openalex"])
    
    # Filtros
    date_range: Dict[str, int] = field(default_factory=lambda: {"from": 2020, "to": 2026})
    venues: List[str] = field(default_factory=list)
    min_citations: int = 0
    
    # Classificacao
    categories: List[str] = field(default_factory=list)
    
    # Output
    output_dir: str = "./systematic_review_output"


@dataclass
class Paper:
    """Representa um artigo encontrado."""
    title: str
    authors: List[str]
    year: int
    abstract: str
    source: str  # arxiv, openalex, etc.
    source_id: str
    url: str
    doi: Optional[str] = None
    venue: Optional[str] = None
    citations: int = 0
    categories: List[str] = field(default_factory=list)
    
    # Campos da revisao
    included: Optional[bool] = None
    exclusion_reason: Optional[str] = None
    review_category: Optional[str] = None
    notes: str = ""
    
    def to_dict(self):
        return asdict(self)


class ArXivSearcher:
    """Busca artigos no arXiv."""
    
    API_URL = "http://export.arxiv.org/api/query"
    CATEGORIES = ["cs.AI", "cs.CL", "cs.LG", "cs.CR", "cs.CV", "stat.ML"]
    
    def search(self, query: str, max_results: int = 100, start: int = 0) -> List[Paper]:
        """Busca artigos no arXiv."""
        query_encoded = query.replace(" ", "+")
        cat_filter = " OR ".join([f"cat:{cat}" for cat in self.CATEGORIES])
        full_query = f"all:{query_encoded} AND ({cat_filter})"
        
        url = (
            f"{self.API_URL}?"
            f"search_query={full_query}"
            f"&start={start}&max_results={max_results}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return self._parse_response(response.content, query)
        except Exception as e:
            print(f"Erro ao buscar no arXiv: {e}")
            return []
    
    def _parse_response(self, content: bytes, source_query: str) -> List[Paper]:
        """Parseia a resposta XML do arXiv."""
        papers = []
        root = ET.fromstring(content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        for entry in root.findall('atom:entry', ns):
            title_elem = entry.find('atom:title', ns)
            summary_elem = entry.find('atom:summary', ns)
            published_elem = entry.find('atom:published', ns)
            link_elem = entry.find('atom:id', ns)
            
            title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else "N/A"
            summary = summary_elem.text.strip() if summary_elem is not None else ""
            published = published_elem.text if published_elem is not None else ""
            link = link_elem.text if link_elem is not None else ""
            
            authors = []
            for author in entry.findall('atom:author', ns):
                name_elem = author.find('atom:name', ns)
                if name_elem is not None:
                    authors.append(name_elem.text)
            
            categories = []
            for cat in entry.findall('atom:category', ns):
                term = cat.get('term', '')
                if term:
                    categories.append(term)
            
            year = int(published[:4]) if published else 0
            
            papers.append(Paper(
                title=title,
                authors=authors,
                year=year,
                abstract=summary,
                source="arxiv",
                source_id=link.split('/')[-1],
                url=link,
                categories=categories,
            ))
        
        return papers


class OpenAlexSearcher:
    """Busca artigos no OpenAlex."""
    
    API_URL = "https://api.openalex.org/works"
    
    def search(self, query: str, max_results: int = 100, from_year: int = 2020, to_year: int = 2026) -> List[Paper]:
        """Busca artigos no OpenAlex."""
        params = {
            "search": query,
            "filter": f"publication_year:{from_year}|{to_year}",
            "per-page": min(max_results, 200),
            "sort": "cited_by_count:desc",
            "mailto": "xuehermes@gmail.com",
        }
        
        try:
            response = requests.get(self.API_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return self._parse_response(data.get("results", []), query)
        except Exception as e:
            print(f"Erro ao buscar no OpenAlex: {e}")
            return []
    
    def _parse_response(self, results: List[Dict], source_query: str) -> List[Paper]:
        """Parseia a resposta JSON do OpenAlex."""
        papers = []
        
        for work in results:
            # Parse abstract
            abstract_idx = work.get("abstract_inverted_index", {})
            abstract = self._format_abstract(abstract_idx)
            
            # Parse authors
            authors_list = work.get("authorships", []) or []
            authors = [a.get("author", {}).get("display_name", "") for a in authors_list]
            
            # Parse venue
            venues = work.get("host_venue", {}) or {}
            venue = venues.get("display_name", "")
            
            papers.append(Paper(
                title=work.get("display_name", "N/A"),
                authors=authors,
                year=work.get("publication_year", 0),
                abstract=abstract,
                source="openalex",
                source_id=work.get("id", "").split("/")[-1],
                url=work.get("id", ""),
                doi=work.get("doi", ""),
                venue=venue,
                citations=work.get("cited_by_count", 0),
            ))
        
        return papers
    
    def _format_abstract(self, abstract_inverted_index: Optional[Dict]) -> str:
        """Converte abstract do formato inverted index para texto."""
        if not abstract_inverted_index:
            return ""
        
        word_positions = []
        for word, positions in abstract_inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        
        word_positions.sort(key=lambda x: x[0])
        return " ".join([w for _, w in word_positions])


class SystematicReview:
    """Orquestra a revisao sistematica."""
    
    def __init__(self, config: ReviewConfig):
        self.config = config
        self.papers: List[Paper] = []
        self.searchers = {
            "arxiv": ArXivSearcher(),
            "openalex": OpenAlexSearcher(),
        }
    
    def run_search(self) -> List[Paper]:
        """Executa a busca em todas as fontes configuradas."""
        print(f"\n{'='*60}")
        print(f"REVISAO SISTEMATICA: {self.config.title}")
        print(f"{'='*60}\n")
        
        all_papers = []
        
        for source in self.config.sources:
            if source not in self.searchers:
                print(f"Fonte desconhecida: {source}")
                continue
            
            searcher = self.searchers[source]
            print(f"\nBuscando em: {source.upper()}")
            
            for query in self.config.search_queries:
                print(f"  Query: {query}")
                
                if source == "openalex":
                    papers = searcher.search(
                        query,
                        from_year=self.config.date_range["from"],
                        to_year=self.config.date_range["to"],
                    )
                else:
                    papers = searcher.search(query)
                
                print(f"    Encontrados: {len(papers)}")
                all_papers.extend(papers)
                
                # Respeita rate limits
                time.sleep(1)
        
        # Remove duplicatas
        seen_ids = set()
        unique_papers = []
        for p in all_papers:
            key = f"{p.source}:{p.source_id}"
            if key not in seen_ids:
                seen_ids.add(key)
                unique_papers.append(p)
        
        self.papers = unique_papers
        print(f"\n{'='*60}")
        print(f"Total de artigos unicos: {len(self.papers)}")
        print(f"{'='*60}")
        
        return self.papers
    
    def apply_screening(self) -> List[Paper]:
        """Aplica criterios de inclusao/exclusao automaticos."""
        print("\nAplicando screening automatico...")
        
        included = []
        excluded = []
        
        for paper in self.papers:
            # Verifica criterios de exclusao
            exclusion_reason = self._check_exclusion(paper)
            if exclusion_reason:
                paper.included = False
                paper.exclusion_reason = exclusion_reason
                excluded.append(paper)
                continue
            
            # Verifica criterios de inclusao
            if self._check_inclusion(paper):
                paper.included = True
                included.append(paper)
            else:
                paper.included = False
                paper.exclusion_reason = "Nao atende criterios de inclusao"
                excluded.append(paper)
        
        print(f"  Incluidos: {len(included)}")
        print(f"  Excluidos: {len(excluded)}")
        
        self.papers = included + excluded
        return included
    
    def _check_exclusion(self, paper: Paper) -> Optional[str]:
        """Verifica se o artigo deve ser excluido."""
        text = f"{paper.title} {paper.abstract}".lower()
        
        for criterion in self.config.exclusion_criteria:
            # Se o criterio comecar com "!", e uma palavra-chave de exclusao
            if criterion.startswith("!"):
                keyword = criterion[1:].lower()
                if keyword in text:
                    return f"Contem termo excluido: {keyword}"
        
        # Filtro de ano
        if paper.year < self.config.date_range["from"] or paper.year > self.config.date_range["to"]:
            return f"Fora do intervalo de datas: {paper.year}"
        
        # Filtro de citacoes
        if paper.citations < self.config.min_citations:
            return f"Citations insuficientes: {paper.citations}"
        
        return None
    
    def _check_inclusion(self, paper: Paper) -> bool:
        """Verifica se o artigo atende criterios de inclusao."""
        text = f"{paper.title} {paper.abstract}".lower()
        
        # Deve atender pelo menos um criterio de inclusao
        for criterion in self.config.inclusion_criteria:
            if criterion.lower() in text:
                return True
        
        return False
    
    def classify_papers(self) -> None:
        """Classifica artigos incluidos em categorias."""
        if not self.config.categories:
            return
        
        print("\nClassificando artigos...")
        
        for paper in self.papers:
            if not paper.included:
                continue
            
            text = f"{paper.title} {paper.abstract}".lower()
            
            for category in self.config.categories:
                if category.lower() in text:
                    paper.review_category = category
                    break
    
    def export_results(self) -> None:
        """Exporta resultados em multiplos formatos."""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Exporta todos os artigos (JSON)
        all_papers_file = output_path / f"all_papers_{timestamp}.json"
        with open(all_papers_file, 'w', encoding='utf-8') as f:
            json.dump([p.to_dict() for p in self.papers], f, indent=2, ensure_ascii=False)
        
        # Exporta artigos incluidos (JSON)
        included = [p for p in self.papers if p.included]
        included_file = output_path / f"included_papers_{timestamp}.json"
        with open(included_file, 'w', encoding='utf-8') as f:
            json.dump([p.to_dict() for p in included], f, indent=2, ensure_ascii=False)
        
        # Exporta artigos excluidos (JSON)
        excluded = [p for p in self.papers if p.included is False]
        excluded_file = output_path / f"excluded_papers_{timestamp}.json"
        with open(excluded_file, 'w', encoding='utf-8') as f:
            json.dump([p.to_dict() for p in excluded], f, indent=2, ensure_ascii=False)
        
        # Exporta CSV para analise em Excel
        csv_file = output_path / f"review_results_{timestamp}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if included:
                writer = csv.DictWriter(f, fieldnames=included[0].to_dict().keys())
                writer.writeheader()
                for p in included:
                    writer.writerow(p.to_dict())
        
        # Gera relatorio PRISMA
        self._generate_prisma_report(output_path, timestamp, included, excluded)
        
        print(f"\nResultados exportados para: {output_path}")
        print(f"  - all_papers_{timestamp}.json")
        print(f"  - included_papers_{timestamp}.json")
        print(f"  - excluded_papers_{timestamp}.json")
        print(f"  - review_results_{timestamp}.csv")
        print(f"  - prisma_report_{timestamp}.md")
    
    def _generate_prisma_report(self, output_path: Path, timestamp: str, 
                                 included: List[Paper], excluded: List[Paper]) -> None:
        """Gera relatorio no formato PRISMA."""
        
        # Contagem por fonte
        source_counts = {}
        for p in self.papers:
            source_counts[p.source] = source_counts.get(p.source, 0) + 1
        
        # Contagem por razao de exclusao
        exclusion_reasons = {}
        for p in excluded:
            reason = p.exclusion_reason or "Desconhecido"
            exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1
        
        # Contagem por categoria
        category_counts = {}
        for p in included:
            cat = p.review_category or "Nao classificado"
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        report = f"""# Relatorio PRISMA - Revisao Sistematica

## {self.config.title}

**Data da busca:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Pergunta de pesquisa:** {self.config.research_question}

---

## Fluxo PRISMA

### Identificacao
- Artigos encontrados nas fontes:
"""
        
        for source, count in source_counts.items():
            report += f"  - {source}: {count}\n"
        
        report += f"""
- **Total identificado:** {len(self.papers)}

### Screening
- Artigos apos remocao de duplicatas: {len(self.papers)}
- Artigos excluidos no screening: {len(excluded)}

### Elegibilidade
- Artigos incluidos na revisao: {len(included)}

---

## Criterios de Inclusao
"""
        for i, criterion in enumerate(self.config.inclusion_criteria, 1):
            report += f"{i}. {criterion}\n"
        
        report += "\n## Criterios de Exclusao\n"
        for i, criterion in enumerate(self.config.exclusion_criteria, 1):
            report += f"{i}. {criterion}\n"
        
        report += f"""
## Razoes de Exclusao
"""
        for reason, count in sorted(exclusion_reasons.items(), key=lambda x: x[1], reverse=True):
            report += f"- {reason}: {count}\n"
        
        report += f"""
## Distribuicao por Categoria
"""
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            report += f"- {cat}: {count}\n"
        
        report += f"""
## Artigos Incluidos ({len(included)})

"""
        for i, paper in enumerate(included, 1):
            report += f"""### {i}. {paper.title}
- **Autores:** {', '.join(paper.authors[:3])}{' et al.' if len(paper.authors) > 3 else ''}
- **Ano:** {paper.year}
- **Fonte:** {paper.source}
- **Categoria:** {paper.review_category or 'Nao classificada'}
- **URL:** {paper.url}

"""
        
        report_file = output_path / f"prisma_report_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)


def create_review_template(name: str) -> Path:
    """Cria um template de configuracao para uma nova revisao."""
    template = {
        "title": f"Revisao Sistematica: {name}",
        "research_question": "Qual e a pergunta de pesquisa principal?",
        "objectives": "Quais sao os objetivos desta revisao?",
        "inclusion_criteria": [
            "artigos sobre language models",
            "artigos sobre interpretabilidade",
            "artigos publicados em conferencias ou revistas",
        ],
        "exclusion_criteria": [
            "!survey",  # Exclui surveys (comecam com !)
            "!review",
            "artigos nao peer-reviewed",
        ],
        "search_queries": [
            "mechanistic interpretability",
            "sparse autoencoder",
            "activation patching",
        ],
        "sources": ["arxiv", "openalex"],
        "date_range": {"from": 2020, "to": 2026},
        "venues": [],
        "min_citations": 0,
        "categories": [
            "interpretabilidade",
            "seguranca",
            "alinhamento",
            "adversarial",
        ],
        "output_dir": f"./{name}_output",
    }
    
    config_path = Path(f"{name}_config.yaml")
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(template, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Template criado: {config_path}")
    print("Edite o arquivo YAML e depois execute:")
    print(f"  python systematic_review.py --run {config_path}")
    
    return config_path


def main():
    parser = argparse.ArgumentParser(description="Ferramenta de Revisao Sistematica da Literatura")
    parser.add_argument("--init", metavar="NAME", help="Cria template de configuracao")
    parser.add_argument("--run", metavar="CONFIG", help="Executa revisao com configuracao")
    parser.add_argument("--config", metavar="CONFIG", help="Alias para --run")
    
    args = parser.parse_args()
    
    if args.init:
        create_review_template(args.init)
        return
    
    config_file = args.run or args.config
    if not config_file:
        parser.print_help()
        return
    
    # Carrega configuracao
    with open(config_file, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    config = ReviewConfig(**config_data)
    
    # Executa revisao
    review = SystematicReview(config)
    review.run_search()
    review.apply_screening()
    review.classify_papers()
    review.export_results()
    
    print("\nRevisao sistematica concluida!")


if __name__ == '__main__':
    main()
