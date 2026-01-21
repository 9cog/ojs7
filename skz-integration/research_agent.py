"""
Research Discovery Agent Implementation
Specialized agent for literature discovery, trend analysis, and research gap identification
"""

import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import requests
from src.models.agent import Agent, AgentCapability, MessageType, db
import openai
import os

class ResearchDiscoveryAgent(Agent):
    """
    Research Discovery Agent for automated literature discovery and analysis
    Implements advanced AI capabilities for research trend identification
    """
    
    def __init__(self, name: str = "Research Discovery Agent"):
        super().__init__(
            name=name,
            agent_type="research_discovery",
            capabilities=[AgentCapability.RESEARCH_DISCOVERY],
            arena_context={
                "databases": ["pubmed", "arxiv", "semantic_scholar", "crossref"],
                "domains": ["computer_science", "medicine", "physics", "biology"],
                "analysis_types": ["trend_analysis", "gap_identification", "impact_assessment"]
            }
        )
        
        # Initialize OpenAI client if API key is available
        self.openai_client = None
        if os.getenv('OPENAI_API_KEY'):
            self.openai_client = openai.OpenAI()
        
        # Research databases and APIs
        self.databases = {
            "semantic_scholar": "https://api.semanticscholar.org/graph/v1",
            "crossref": "https://api.crossref.org/works",
            "arxiv": "http://export.arxiv.org/api/query"
        }
        
        # Cache for research results
        self.research_cache = {}
    
    def _process_message(self, message):
        """Process research-related messages"""
        try:
            content = json.loads(message.content) if isinstance(message.content, str) else message.content
            
            if message.message_type == MessageType.QUERY:
                response_data = self._handle_research_query(content)
            elif message.message_type == MessageType.COMMAND:
                response_data = self._handle_research_command(content)
            else:
                response_data = {"error": "Unsupported message type"}
            
            # Send response back
            self.send_message(
                recipient_id=message.sender_id,
                message_type=MessageType.RESPONSE,
                content=response_data,
                correlation_id=message.id
            )
            
            # Update performance metrics
            metrics = self.get_performance_metrics()
            metrics['tasks_completed'] += 1
            self.update_performance_metrics(metrics)
            
        except Exception as e:
            error_response = {"error": str(e), "agent": self.name}
            self.send_message(
                recipient_id=message.sender_id,
                message_type=MessageType.RESPONSE,
                content=error_response,
                correlation_id=message.id
            )
    
    def _handle_research_query(self, content: Dict) -> Dict:
        """Handle research query requests"""
        query_type = content.get('query_type')
        
        if query_type == 'literature_search':
            return self._perform_literature_search(content)
        elif query_type == 'trend_analysis':
            return self._perform_trend_analysis(content)
        elif query_type == 'gap_identification':
            return self._identify_research_gaps(content)
        elif query_type == 'impact_assessment':
            return self._assess_research_impact(content)
        else:
            return {"error": f"Unknown query type: {query_type}"}
    
    def _handle_research_command(self, content: Dict) -> Dict:
        """Handle research command requests"""
        command = content.get('command')
        
        if command == 'update_knowledge_base':
            return self._update_knowledge_base(content)
        elif command == 'generate_research_report':
            return self._generate_research_report(content)
        else:
            return {"error": f"Unknown command: {command}"}
    
    def _perform_literature_search(self, params: Dict) -> Dict:
        """Perform comprehensive literature search across multiple databases"""
        query = params.get('query', '')
        domain = params.get('domain', 'computer_science')
        limit = params.get('limit', 20)
        
        # Check cache first
        cache_key = f"search_{hash(query)}_{domain}_{limit}"
        if cache_key in self.research_cache:
            cached_result = self.research_cache[cache_key]
            if datetime.now() - cached_result['timestamp'] < timedelta(hours=1):
                return cached_result['data']
        
        results = {
            "query": query,
            "domain": domain,
            "papers": [],
            "total_found": 0,
            "search_timestamp": datetime.now().isoformat()
        }
        
        try:
            # Search Semantic Scholar
            semantic_results = self._search_semantic_scholar(query, limit)
            results["papers"].extend(semantic_results)
            
            # Search arXiv for CS papers
            if domain in ['computer_science', 'physics', 'mathematics']:
                arxiv_results = self._search_arxiv(query, limit)
                results["papers"].extend(arxiv_results)
            
            # Remove duplicates and sort by relevance
            results["papers"] = self._deduplicate_papers(results["papers"])
            results["total_found"] = len(results["papers"])
            
            # Enhance with AI analysis if available
            if self.openai_client and results["papers"]:
                results["ai_analysis"] = self._analyze_papers_with_ai(results["papers"][:5])
            
            # Cache results
            self.research_cache[cache_key] = {
                'data': results,
                'timestamp': datetime.now()
            }
            
            return results
            
        except Exception as e:
            return {"error": f"Literature search failed: {str(e)}"}
    
    def _search_semantic_scholar(self, query: str, limit: int) -> List[Dict]:
        """Search Semantic Scholar API"""
        try:
            url = f"{self.databases['semantic_scholar']}/paper/search"
            params = {
                'query': query,
                'limit': limit,
                'fields': 'title,authors,year,abstract,citationCount,url,venue'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            papers = []
            
            for paper in data.get('data', []):
                papers.append({
                    'title': paper.get('title', ''),
                    'authors': [author.get('name', '') for author in paper.get('authors', [])],
                    'year': paper.get('year'),
                    'abstract': paper.get('abstract', ''),
                    'citation_count': paper.get('citationCount', 0),
                    'url': paper.get('url', ''),
                    'venue': paper.get('venue', ''),
                    'source': 'semantic_scholar'
                })
            
            return papers
            
        except Exception as e:
            print(f"Semantic Scholar search error: {e}")
            return []
    
    def _search_arxiv(self, query: str, limit: int) -> List[Dict]:
        """Search arXiv API"""
        try:
            url = self.databases['arxiv']
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': limit,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            # Parse XML response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            papers = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
                summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
                published = entry.find('{http://www.w3.org/2005/Atom}published').text
                
                authors = []
                for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                    name = author.find('{http://www.w3.org/2005/Atom}name').text
                    authors.append(name)
                
                papers.append({
                    'title': title,
                    'authors': authors,
                    'year': int(published[:4]),
                    'abstract': summary,
                    'citation_count': 0,  # arXiv doesn't provide citation counts
                    'url': entry.find('{http://www.w3.org/2005/Atom}id').text,
                    'venue': 'arXiv',
                    'source': 'arxiv'
                })
            
            return papers
            
        except Exception as e:
            print(f"arXiv search error: {e}")
            return []
    
    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicate papers based on title similarity"""
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            title = paper.get('title', '').lower().strip()
            # Simple deduplication based on title
            title_key = re.sub(r'[^\w\s]', '', title)
            
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_papers.append(paper)
        
        # Sort by citation count and year
        unique_papers.sort(key=lambda x: (x.get('citation_count', 0), x.get('year', 0)), reverse=True)
        return unique_papers
    
    def _analyze_papers_with_ai(self, papers: List[Dict]) -> Dict:
        """Use AI to analyze papers and extract insights"""
        if not self.openai_client:
            return {"error": "OpenAI client not available"}
        
        try:
            # Prepare paper summaries for AI analysis
            paper_summaries = []
            for paper in papers:
                summary = f"Title: {paper.get('title', '')}\n"
                summary += f"Authors: {', '.join(paper.get('authors', []))}\n"
                summary += f"Year: {paper.get('year', 'Unknown')}\n"
                summary += f"Abstract: {paper.get('abstract', '')[:500]}...\n"
                paper_summaries.append(summary)
            
            prompt = f"""
            Analyze the following research papers and provide insights:
            
            {chr(10).join(paper_summaries)}
            
            Please provide:
            1. Key themes and trends
            2. Research gaps identified
            3. Methodological approaches
            4. Potential future directions
            5. Overall assessment of the field
            
            Format your response as JSON with these keys: themes, gaps, methods, future_directions, assessment
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            
            # Parse AI response
            ai_content = response.choices[0].message.content
            try:
                return json.loads(ai_content)
            except json.JSONDecodeError:
                return {"ai_analysis": ai_content}
                
        except Exception as e:
            return {"error": f"AI analysis failed: {str(e)}"}
    
    def _perform_trend_analysis(self, params: Dict) -> Dict:
        """Analyze research trends in a specific domain"""
        domain = params.get('domain', 'computer_science')
        time_range = params.get('time_range', 5)  # years
        
        # This would typically involve more sophisticated analysis
        # For now, return a simplified trend analysis
        return {
            "domain": domain,
            "time_range": time_range,
            "trends": [
                {
                    "topic": "Machine Learning",
                    "growth_rate": 0.25,
                    "paper_count": 1500,
                    "key_keywords": ["neural networks", "deep learning", "AI"]
                },
                {
                    "topic": "Natural Language Processing",
                    "growth_rate": 0.30,
                    "paper_count": 800,
                    "key_keywords": ["NLP", "transformers", "language models"]
                }
            ],
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _identify_research_gaps(self, params: Dict) -> Dict:
        """Identify potential research gaps in a domain"""
        domain = params.get('domain', 'computer_science')
        
        # Simplified gap identification
        return {
            "domain": domain,
            "identified_gaps": [
                {
                    "gap_area": "Explainable AI in Healthcare",
                    "description": "Limited research on interpretable ML models for medical diagnosis",
                    "opportunity_score": 0.85,
                    "related_keywords": ["explainable AI", "healthcare", "interpretability"]
                },
                {
                    "gap_area": "Sustainable Computing",
                    "description": "Insufficient focus on energy-efficient algorithms",
                    "opportunity_score": 0.75,
                    "related_keywords": ["green computing", "energy efficiency", "sustainability"]
                }
            ],
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _assess_research_impact(self, params: Dict) -> Dict:
        """Assess the potential impact of research topics"""
        topics = params.get('topics', [])
        
        impact_assessment = []
        for topic in topics:
            # Simplified impact assessment
            impact_assessment.append({
                "topic": topic,
                "impact_score": 0.8,  # Would be calculated based on various factors
                "factors": {
                    "novelty": 0.7,
                    "applicability": 0.9,
                    "market_potential": 0.8,
                    "academic_interest": 0.85
                },
                "recommendation": "High potential for significant impact"
            })
        
        return {
            "impact_assessments": impact_assessment,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _update_knowledge_base(self, params: Dict) -> Dict:
        """Update the agent's knowledge base with new information"""
        # This would typically update a vector database or knowledge graph
        return {
            "status": "knowledge_base_updated",
            "updated_entries": params.get('entries', 0),
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_research_report(self, params: Dict) -> Dict:
        """Generate a comprehensive research report"""
        topic = params.get('topic', 'General Research')
        
        # This would generate a detailed report based on collected research
        return {
            "report_title": f"Research Report: {topic}",
            "generated_at": datetime.now().isoformat(),
            "sections": [
                {
                    "title": "Executive Summary",
                    "content": "Overview of current research landscape..."
                },
                {
                    "title": "Key Findings",
                    "content": "Major discoveries and trends identified..."
                },
                {
                    "title": "Research Gaps",
                    "content": "Areas requiring further investigation..."
                },
                {
                    "title": "Recommendations",
                    "content": "Suggested research directions..."
                }
            ]
        }

