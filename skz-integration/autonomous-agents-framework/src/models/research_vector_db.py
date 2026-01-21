"""
Unified Research Vector Database for Research Discovery Agent
Advanced semantic search and knowledge graph for academic research
"""
import asyncio
import logging
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
from collections import defaultdict
import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

@dataclass
class ResearchDocument:
    """Research document structure for vector storage"""
    doc_id: str
    title: str
    abstract: str
    authors: List[str]
    keywords: List[str]
    doi: str
    publication_date: str
    journal: str
    citation_count: int
    full_text: str
    document_type: str
    research_areas: List[str]
    methodology: str
    findings: List[str]
    significance_score: float
    novelty_score: float
    embedding: Optional[List[float]] = None

@dataclass
class ResearchQuery:
    """Research query structure"""
    query_text: str
    research_areas: List[str]
    date_range: Optional[Tuple[str, str]]
    min_citation_count: int
    document_types: List[str]
    methodology_filter: List[str]
    similarity_threshold: float

@dataclass
class ResearchResult:
    """Research search result"""
    document: ResearchDocument
    similarity_score: float
    relevance_reasons: List[str]
    trend_indicators: List[str]
    related_documents: List[str]

class ResearchVectorDB:
    """Unified research vector database with semantic search"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=config.get('db_path', './research_vectordb'))
        self.collection = self.client.get_or_create_collection(
            name="research_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Research area hierarchies
        self.research_hierarchies = {
            'cosmetic_chemistry': [
                'formulation_science', 'ingredient_safety', 'stability_testing',
                'emulsification', 'rheology', 'preservation'
            ],
            'dermatology': [
                'skin_biology', 'dermatitis', 'skin_barrier', 'wound_healing',
                'cosmetic_dermatology', 'sensitive_skin'
            ],
            'materials_science': [
                'nanotechnology', 'biomaterials', 'surface_science',
                'polymer_science', 'crystallography'
            ],
            'analytical_chemistry': [
                'chromatography', 'spectroscopy', 'mass_spectrometry',
                'method_validation', 'quality_control'
            ],
            'toxicology': [
                'safety_assessment', 'risk_assessment', 'in_vitro_testing',
                'alternative_methods', 'regulatory_toxicology'
            ]
        }
        
    async def add_document(self, document: ResearchDocument) -> bool:
        """Add research document to vector database"""
        try:
            # Generate embedding if not provided
            if document.embedding is None:
                text_for_embedding = f"{document.title} {document.abstract} {' '.join(document.keywords)}"
                document.embedding = self.model.encode(text_for_embedding).tolist()
            
            # Prepare metadata
            metadata = {
                'title': document.title,
                'authors': json.dumps(document.authors),
                'keywords': json.dumps(document.keywords),
                'doi': document.doi,
                'publication_date': document.publication_date,
                'journal': document.journal,
                'citation_count': document.citation_count,
                'document_type': document.document_type,
                'research_areas': json.dumps(document.research_areas),
                'methodology': document.methodology,
                'significance_score': document.significance_score,
                'novelty_score': document.novelty_score
            }
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=[document.embedding],
                documents=[f"{document.title}\n\n{document.abstract}"],
                metadatas=[metadata],
                ids=[document.doc_id]
            )
            
            logger.info(f"Added document {document.doc_id} to vector database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document to vector database: {e}")
            return False
    
    async def search_research(self, query: ResearchQuery, limit: int = 20) -> List[ResearchResult]:
        """Search research documents using semantic similarity"""
        try:
            # Generate query embedding
            query_embedding = self.model.encode(query.query_text).tolist()
            
            # Build metadata filters
            where_filters = {}
            
            if query.research_areas:
                # This would need to be handled differently in ChromaDB
                pass
            
            if query.min_citation_count > 0:
                where_filters['citation_count'] = {'$gte': query.min_citation_count}
            
            if query.document_types:
                where_filters['document_type'] = {'$in': query.document_types}
            
            # Perform vector search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_filters if where_filters else None
            )
            
            # Process results
            search_results = []
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                distance = results['distances'][0][i]
                similarity = 1 - distance  # Convert distance to similarity
                
                if similarity >= query.similarity_threshold:
                    # Reconstruct document from metadata
                    metadata = results['metadatas'][0][i]
                    document = ResearchDocument(
                        doc_id=doc_id,
                        title=metadata['title'],
                        abstract=results['documents'][0][i].split('\n\n', 1)[1] if '\n\n' in results['documents'][0][i] else '',
                        authors=json.loads(metadata['authors']),
                        keywords=json.loads(metadata['keywords']),
                        doi=metadata['doi'],
                        publication_date=metadata['publication_date'],
                        journal=metadata['journal'],
                        citation_count=metadata['citation_count'],
                        full_text='',  # Not stored in vector DB
                        document_type=metadata['document_type'],
                        research_areas=json.loads(metadata['research_areas']),
                        methodology=metadata['methodology'],
                        findings=[],  # Would need separate storage
                        significance_score=metadata['significance_score'],
                        novelty_score=metadata['novelty_score']
                    )
                    
                    # Generate relevance reasoning
                    relevance_reasons = await self._generate_relevance_reasons(query, document, similarity)
                    
                    # Identify trend indicators
                    trend_indicators = await self._identify_trend_indicators(document)
                    
                    # Find related documents
                    related_docs = await self._find_related_documents(doc_id, limit=5)
                    
                    result = ResearchResult(
                        document=document,
                        similarity_score=similarity,
                        relevance_reasons=relevance_reasons,
                        trend_indicators=trend_indicators,
                        related_documents=related_docs
                    )
                    
                    search_results.append(result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching research database: {e}")
            return []
    
    async def find_research_trends(self, research_area: str, time_window: int = 24) -> Dict[str, Any]:
        """Identify research trends in specific area"""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date.replace(month=end_date.month - time_window)
            
            # Search for recent papers in area
            query = ResearchQuery(
                query_text=research_area,
                research_areas=[research_area],
                date_range=(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')),
                min_citation_count=0,
                document_types=['research_article', 'review'],
                methodology_filter=[],
                similarity_threshold=0.6
            )
            
            recent_papers = await self.search_research(query, limit=100)
            
            # Analyze trends
            keyword_freq = defaultdict(int)
            methodology_trends = defaultdict(int)
            citation_trends = defaultdict(list)
            
            for result in recent_papers:
                doc = result.document
                
                # Keyword analysis
                for keyword in doc.keywords:
                    keyword_freq[keyword.lower()] += 1
                
                # Methodology trends
                if doc.methodology:
                    methodology_trends[doc.methodology] += 1
                
                # Citation trends by month
                pub_date = datetime.strptime(doc.publication_date, '%Y-%m-%d')
                month_key = pub_date.strftime('%Y-%m')
                citation_trends[month_key].append(doc.citation_count)
            
            # Generate trend summary
            trending_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            trending_methods = sorted(methodology_trends.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Calculate growth metrics
            total_papers = len(recent_papers)
            avg_citations = sum(r.document.citation_count for r in recent_papers) / total_papers if total_papers > 0 else 0
            avg_novelty = sum(r.document.novelty_score for r in recent_papers) / total_papers if total_papers > 0 else 0
            
            return {
                'research_area': research_area,
                'time_window_months': time_window,
                'total_papers': total_papers,
                'trending_keywords': trending_keywords,
                'trending_methodologies': trending_methods,
                'average_citations': avg_citations,
                'average_novelty_score': avg_novelty,
                'growth_indicators': self._calculate_growth_indicators(citation_trends),
                'emerging_topics': await self._identify_emerging_topics(recent_papers),
                'research_gaps': await self._identify_research_gaps(research_area, recent_papers)
            }
            
        except Exception as e:
            logger.error(f"Error finding research trends: {e}")
            return {}
    
    async def build_knowledge_graph(self, documents: List[ResearchDocument]) -> Dict[str, Any]:
        """Build knowledge graph from research documents"""
        try:
            nodes = {}
            edges = []
            
            for doc in documents:
                # Add document node
                nodes[doc.doc_id] = {
                    'id': doc.doc_id,
                    'type': 'document',
                    'title': doc.title,
                    'research_areas': doc.research_areas,
                    'citation_count': doc.citation_count,
                    'significance_score': doc.significance_score
                }
                
                # Add author nodes and edges
                for author in doc.authors:
                    author_id = f"author_{hashlib.md5(author.encode()).hexdigest()[:8]}"
                    if author_id not in nodes:
                        nodes[author_id] = {
                            'id': author_id,
                            'type': 'author',
                            'name': author
                        }
                    
                    edges.append({
                        'source': author_id,
                        'target': doc.doc_id,
                        'type': 'authored',
                        'weight': 1.0
                    })
                
                # Add keyword nodes and edges
                for keyword in doc.keywords:
                    keyword_id = f"keyword_{hashlib.md5(keyword.encode()).hexdigest()[:8]}"
                    if keyword_id not in nodes:
                        nodes[keyword_id] = {
                            'id': keyword_id,
                            'type': 'keyword',
                            'term': keyword
                        }
                    
                    edges.append({
                        'source': doc.doc_id,
                        'target': keyword_id,
                        'type': 'contains_keyword',
                        'weight': 1.0
                    })
            
            # Calculate centrality and importance scores
            node_importance = self._calculate_node_importance(nodes, edges)
            
            return {
                'nodes': list(nodes.values()),
                'edges': edges,
                'statistics': {
                    'total_nodes': len(nodes),
                    'total_edges': len(edges),
                    'document_nodes': len([n for n in nodes.values() if n['type'] == 'document']),
                    'author_nodes': len([n for n in nodes.values() if n['type'] == 'author']),
                    'keyword_nodes': len([n for n in nodes.values() if n['type'] == 'keyword'])
                },
                'important_nodes': sorted(node_importance.items(), key=lambda x: x[1], reverse=True)[:20]
            }
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            return {}
    
    async def _generate_relevance_reasons(self, query: ResearchQuery, document: ResearchDocument, similarity: float) -> List[str]:
        """Generate reasons why document is relevant to query"""
        reasons = []
        
        # Similarity-based reasoning
        if similarity > 0.9:
            reasons.append("Very high semantic similarity to query")
        elif similarity > 0.8:
            reasons.append("High semantic similarity to query")
        
        # Research area matching
        query_areas_lower = [area.lower() for area in query.research_areas]
        doc_areas_lower = [area.lower() for area in document.research_areas]
        
        matching_areas = set(query_areas_lower) & set(doc_areas_lower)
        if matching_areas:
            reasons.append(f"Matches research areas: {', '.join(matching_areas)}")
        
        # Keyword matching
        query_words = set(query.query_text.lower().split())
        doc_keywords = set(keyword.lower() for keyword in document.keywords)
        
        matching_keywords = query_words & doc_keywords
        if matching_keywords:
            reasons.append(f"Contains relevant keywords: {', '.join(list(matching_keywords)[:3])}")
        
        # Citation importance
        if document.citation_count > 100:
            reasons.append("Highly cited paper (significant impact)")
        elif document.citation_count > 50:
            reasons.append("Well-cited paper (good impact)")
        
        # Novelty and significance
        if document.novelty_score > 8.0:
            reasons.append("High novelty score (innovative research)")
        if document.significance_score > 8.0:
            reasons.append("High significance score (important findings)")
        
        return reasons[:5]  # Top 5 reasons
    
    async def _identify_trend_indicators(self, document: ResearchDocument) -> List[str]:
        """Identify trend indicators for document"""
        indicators = []
        
        # Recent publication
        pub_date = datetime.strptime(document.publication_date, '%Y-%m-%d')
        months_old = (datetime.now() - pub_date).days / 30
        
        if months_old < 6:
            indicators.append("Very recent publication")
        elif months_old < 12:
            indicators.append("Recent publication")
        
        # Rapid citation growth
        citation_rate = document.citation_count / max(months_old, 1)
        if citation_rate > 10:
            indicators.append("Rapid citation growth")
        elif citation_rate > 5:
            indicators.append("Good citation momentum")
        
        # Emerging methodology
        emerging_methods = ['ai', 'machine learning', 'deep learning', 'nanotechnology', 'crispr']
        if any(method in document.methodology.lower() for method in emerging_methods):
            indicators.append("Uses emerging methodology")
        
        # Cross-disciplinary research
        if len(document.research_areas) > 2:
            indicators.append("Cross-disciplinary research")
        
        return indicators
    
    async def _find_related_documents(self, doc_id: str, limit: int = 5) -> List[str]:
        """Find documents related to given document"""
        try:
            # Get the document
            result = self.collection.get(ids=[doc_id], include=['embeddings', 'metadatas'])
            
            if not result['ids']:
                return []
            
            # Use document embedding to find similar documents
            doc_embedding = result['embeddings'][0]
            
            similar_results = self.collection.query(
                query_embeddings=[doc_embedding],
                n_results=limit + 1,  # +1 because it will include the original document
                include=['metadatas']
            )
            
            # Return IDs excluding the original document
            related_ids = [id for id in similar_results['ids'][0] if id != doc_id]
            return related_ids[:limit]
            
        except Exception as e:
            logger.error(f"Error finding related documents: {e}")
            return []
    
    def _calculate_growth_indicators(self, citation_trends: Dict[str, List[int]]) -> Dict[str, float]:
        """Calculate growth indicators from citation trends"""
        indicators = {}
        
        if not citation_trends:
            return indicators
        
        # Sort months
        sorted_months = sorted(citation_trends.keys())
        
        if len(sorted_months) >= 2:
            # Calculate month-over-month growth
            recent_avg = np.mean(citation_trends[sorted_months[-1]]) if citation_trends[sorted_months[-1]] else 0
            previous_avg = np.mean(citation_trends[sorted_months[-2]]) if citation_trends[sorted_months[-2]] else 0
            
            if previous_avg > 0:
                mom_growth = (recent_avg - previous_avg) / previous_avg
                indicators['month_over_month_growth'] = mom_growth
            
            # Calculate overall trend slope
            monthly_avgs = []
            for month in sorted_months:
                avg_citations = np.mean(citation_trends[month]) if citation_trends[month] else 0
                monthly_avgs.append(avg_citations)
            
            if len(monthly_avgs) > 1:
                # Simple linear trend
                x = np.arange(len(monthly_avgs))
                trend_slope = np.polyfit(x, monthly_avgs, 1)[0]
                indicators['trend_slope'] = trend_slope
        
        return indicators
    
    async def _identify_emerging_topics(self, recent_papers: List[ResearchResult]) -> List[str]:
        """Identify emerging research topics"""
        # Analyze keyword co-occurrence and novelty
        topic_scores = defaultdict(float)
        
        for result in recent_papers:
            doc = result.document
            
            # Weight by novelty and recency
            weight = doc.novelty_score / 10.0
            
            for keyword in doc.keywords:
                topic_scores[keyword.lower()] += weight
        
        # Return top emerging topics
        emerging = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, score in emerging[:10] if score > 5.0]
    
    async def _identify_research_gaps(self, research_area: str, recent_papers: List[ResearchResult]) -> List[str]:
        """Identify potential research gaps"""
        gaps = []
        
        # Get expected topics for research area
        expected_topics = self.research_hierarchies.get(research_area, [])
        
        # Find which topics are underrepresented
        covered_topics = set()
        for result in recent_papers:
            for keyword in result.document.keywords:
                covered_topics.add(keyword.lower())
        
        for expected_topic in expected_topics:
            if expected_topic not in covered_topics:
                gaps.append(f"Limited research on {expected_topic}")
        
        # Identify methodology gaps
        methodologies_used = set()
        for result in recent_papers:
            if result.document.methodology:
                methodologies_used.add(result.document.methodology.lower())
        
        expected_methods = ['experimental', 'computational', 'clinical', 'meta-analysis']
        for method in expected_methods:
            if not any(method in used_method for used_method in methodologies_used):
                gaps.append(f"Gap in {method} approaches")
        
        return gaps[:5]
    
    def _calculate_node_importance(self, nodes: Dict[str, Any], edges: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate node importance scores"""
        importance = defaultdict(float)
        
        # Calculate degree centrality
        degree_count = defaultdict(int)
        for edge in edges:
            degree_count[edge['source']] += 1
            degree_count[edge['target']] += 1
        
        max_degree = max(degree_count.values()) if degree_count else 1
        
        for node_id, degree in degree_count.items():
            importance[node_id] = degree / max_degree
        
        return dict(importance)


# Utility functions
async def initialize_research_db(config: Dict[str, Any]) -> ResearchVectorDB:
    """Initialize research vector database"""
    return ResearchVectorDB(config)

async def quick_research_search(query_text: str, limit: int = 10) -> List[Dict]:
    """Quick research search utility"""
    config = {'db_path': './research_vectordb'}
    db = ResearchVectorDB(config)
    
    query = ResearchQuery(
        query_text=query_text,
        research_areas=[],
        date_range=None,
        min_citation_count=0,
        document_types=['research_article'],
        methodology_filter=[],
        similarity_threshold=0.6
    )
    
    results = await db.search_research(query, limit)
    return [asdict(result) for result in results]
