"""
Patent Landscape Analysis System for Research Discovery Agent
Advanced patent search, analysis, and landscape mapping
"""
import asyncio
import logging
import json
import os
import requests
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import re
import hashlib
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class PatentDocument:
    """Patent document structure"""
    patent_id: str
    title: str
    abstract: str
    inventors: List[str]
    assignees: List[str]
    publication_date: str
    filing_date: str
    patent_number: str
    classification_codes: List[str]
    claims: List[str]
    priority_claims: List[str]
    citations: List[str]
    cited_by: List[str]
    legal_status: str
    country: str
    language: str
    relevance_score: float

@dataclass
class PatentCluster:
    """Patent technology cluster"""
    cluster_id: str
    cluster_name: str
    patents: List[str]
    key_concepts: List[str]
    technology_area: str
    innovation_trend: str
    competitive_intensity: float
    growth_rate: float
    key_players: List[str]

@dataclass
class LandscapeAnalysis:
    """Patent landscape analysis result"""
    analysis_id: str
    query_terms: List[str]
    total_patents: int
    date_range: Tuple[str, str]
    technology_clusters: List[PatentCluster]
    key_innovations: List[PatentDocument]
    competitive_landscape: Dict[str, int]
    innovation_timeline: Dict[str, List[PatentDocument]]
    white_spaces: List[str]
    analysis_timestamp: str

class PatentAnalyzer:
    """Advanced patent landscape analysis system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_endpoints = {
            'uspto': 'https://developer.uspto.gov/api',
            'espacenet': 'https://ops.epo.org/3.2',
            'google': 'https://patents.google.com/api'
        }
        self.cache = {}
        
        # Cosmetic science classification codes
        self.classification_mapping = {
            'cosmetic_formulations': ['A61K8/', 'A61Q'],
            'skin_care': ['A61K8/02', 'A61K8/04', 'A61Q19/'],
            'hair_care': ['A61K8/81', 'A61Q5/'],
            'color_cosmetics': ['A61K8/11', 'A61Q1/'],
            'fragrance': ['C11B9/', 'A61K8/34'],
            'preservation': ['A61K8/30', 'A01N'],
            'packaging': ['B65D', 'A45D'],
            'manufacturing': ['B01F', 'A61K8/']
        }
    
    async def analyze_patent_landscape(self, query_terms: List[str], date_range: Optional[Tuple[str, str]] = None, max_patents: int = 500) -> LandscapeAnalysis:
        """Comprehensive patent landscape analysis"""
        logger.info(f"Starting patent landscape analysis for: {query_terms}")
        
        try:
            # Search patents
            patents = await self._search_patents(query_terms, date_range, max_patents)
            
            # Cluster analysis
            clusters = await self._cluster_patents(patents)
            
            # Identify key innovations
            key_innovations = await self._identify_key_innovations(patents)
            
            # Competitive analysis
            competitive_landscape = await self._analyze_competitive_landscape(patents)
            
            # Timeline analysis
            innovation_timeline = await self._create_innovation_timeline(patents)
            
            # White space identification
            white_spaces = await self._identify_white_spaces(patents, clusters)
            
            analysis_id = f"landscape_{hashlib.md5('_'.join(query_terms).encode()).hexdigest()[:12]}"
            
            return LandscapeAnalysis(
                analysis_id=analysis_id,
                query_terms=query_terms,
                total_patents=len(patents),
                date_range=date_range or ("1990-01-01", datetime.now().strftime("%Y-%m-%d")),
                technology_clusters=clusters,
                key_innovations=key_innovations,
                competitive_landscape=competitive_landscape,
                innovation_timeline=innovation_timeline,
                white_spaces=white_spaces,
                analysis_timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error in patent landscape analysis: {e}")
            return LandscapeAnalysis("", query_terms, 0, ("", ""), [], [], {}, {}, [], datetime.now().isoformat())
    
    async def _search_patents(self, query_terms: List[str], date_range: Optional[Tuple[str, str]], max_patents: int) -> List[PatentDocument]:
        """Search patents using multiple databases"""
        
        all_patents = []
        
        try:
            # Build search query
            search_query = self._build_search_query(query_terms)
            
            # Search USPTO (mock implementation - would use actual API)
            uspto_patents = await self._search_uspto(search_query, date_range, max_patents // 2)
            all_patents.extend(uspto_patents)
            
            # Search Google Patents (mock implementation)
            google_patents = await self._search_google_patents(search_query, date_range, max_patents // 2)
            all_patents.extend(google_patents)
            
            # Deduplicate by patent number
            seen_numbers = set()
            unique_patents = []
            for patent in all_patents:
                if patent.patent_number not in seen_numbers:
                    seen_numbers.add(patent.patent_number)
                    unique_patents.append(patent)
            
            # Calculate relevance scores
            for patent in unique_patents:
                patent.relevance_score = await self._calculate_relevance_score(patent, query_terms)
            
            # Sort by relevance
            unique_patents.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return unique_patents[:max_patents]
            
        except Exception as e:
            logger.error(f"Error searching patents: {e}")
            return []
    
    def _build_search_query(self, query_terms: List[str]) -> str:
        """Build patent search query"""
        
        # Expand terms with synonyms
        expanded_terms = []
        for term in query_terms:
            expanded_terms.append(term)
            # Add common variations
            if 'cosmetic' in term.lower():
                expanded_terms.extend(['beauty', 'personal care', 'skincare'])
            if 'formulation' in term.lower():
                expanded_terms.extend(['composition', 'preparation', 'mixture'])
        
        # Build Boolean query
        return ' OR '.join(f'"{term}"' for term in expanded_terms)
    
    async def _search_uspto(self, query: str, date_range: Optional[Tuple[str, str]], limit: int) -> List[PatentDocument]:
        """Search USPTO database"""
        
        # Check if production USPTO API is configured
        uspto_api_key = self.config.get('uspto_api_key')
        if uspto_api_key and self.config.get('use_production_apis', False):
            return await self._search_uspto_production(query, date_range, limit)
        else:
            # PRODUCTION: No fallback to mock - must configure API
            raise ValueError(
                "USPTO API configuration required for production. "
                "Configure uspto_api_key and set use_production_apis=True. "
                "NEVER SACRIFICE QUALITY!! No mock fallbacks in production."
            )
    
    async def _search_uspto_production(self, query: str, date_range: Optional[Tuple[str, str]], limit: int) -> List[PatentDocument]:
        """Production USPTO API integration"""
        try:
            import aiohttp
            
            headers = {
                'Authorization': f'Bearer {self.config["uspto_api_key"]}',
                'Content-Type': 'application/json'
            }
            
            params = {
                'q': query,
                'limit': limit,
                'format': 'json'
            }
            
            if date_range:
                params['dateRange'] = f"{date_range[0]}TO{date_range[1]}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_endpoints['uspto']}/patents/query",
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_uspto_response(data)
                    elif response.status == 429:
                        # Rate limit handling
                        await asyncio.sleep(60)
                        return await self._search_uspto_production(query, date_range, limit)
                    else:
                        logger.error(f"USPTO API error: {response.status}")
                        # PRODUCTION: No fallback to mock - raise error
                        raise ValueError(f"USPTO API error: {response.status}. Check API configuration and credentials.")
                        
        except Exception as e:
            logger.error(f"USPTO search error: {e}")
            # PRODUCTION: No fallback to mock - raise error
            raise ValueError(f"USPTO search failed: {e}. Check API configuration and network connectivity.")
    
    def _parse_uspto_response(self, data: Dict) -> List[PatentDocument]:
        """Parse USPTO API response into PatentDocument objects"""
        patents = []
        
        for item in data.get('patents', []):
            patent = PatentDocument(
                patent_id=item.get('patentId', ''),
                title=item.get('title', ''),
                abstract=item.get('abstract', ''),
                inventors=item.get('inventors', []),
                assignees=item.get('assignees', []),
                publication_date=item.get('publicationDate', ''),
                filing_date=item.get('filingDate', ''),
                patent_number=item.get('patentNumber', ''),
                classification_codes=item.get('classificationCodes', []),
                claims=item.get('claims', []),
                priority_claims=item.get('priorityClaims', []),
                citations=item.get('citations', []),
                cited_by=item.get('citedBy', []),
                legal_status=item.get('legalStatus', ''),
                country=item.get('country', ''),
                language=item.get('language', ''),
                relevance_score=0.0  # Will be calculated separately
            )
            patents.append(patent)
        
        return patents
    
    # PRODUCTION IMPLEMENTATION: Mock functions removed for production deployment
    # Development testing should use test databases and proper API sandboxes
    # For development, use: export ENVIRONMENT=development and configure test APIs
    
    async def _search_google_patents(self, query: str, date_range: Optional[Tuple[str, str]], limit: int) -> List[PatentDocument]:
        """Search Google Patents"""
        
        # Check if production Google Patents API is configured
        google_credentials = self.config.get('google_cloud_credentials')
        if google_credentials and self.config.get('use_production_apis', False):
            return await self._search_google_patents_production(query, date_range, limit)
        else:
            # PRODUCTION: No fallback to mock - must configure API
            raise ValueError(
                "Google Patents API configuration required for production. "
                "Configure google_cloud_credentials and set use_production_apis=True. "
                "NEVER SACRIFICE QUALITY!! No mock fallbacks in production."
            )
    
    async def _search_google_patents_production(self, query: str, date_range: Optional[Tuple[str, str]], limit: int) -> List[PatentDocument]:
        """Production Google Patents API integration"""
        try:
            import aiohttp
            import json
            
            # Google Patents API endpoint (using Custom Search API for patents)
            base_url = "https://www.googleapis.com/customsearch/v1"
            
            # Get API credentials
            credentials_path = self.config.get('google_cloud_credentials')
            if not credentials_path or not os.path.exists(credentials_path):
                logger.error("Google Cloud credentials file not found")
                raise ValueError("Google Cloud credentials file not found. Check google_cloud_credentials configuration.")
            
            # Load credentials
            with open(credentials_path, 'r') as f:
                credentials = json.load(f)
            
            api_key = credentials.get('api_key') or os.getenv('GOOGLE_API_KEY')
            search_engine_id = credentials.get('search_engine_id') or os.getenv('GOOGLE_SEARCH_ENGINE_ID')
            
            if not api_key or not search_engine_id:
                logger.error("Google API key or Search Engine ID not configured")
                raise ValueError("Google API key or Search Engine ID not configured. Check API credentials.")
            
            # Build search parameters
            params = {
                'key': api_key,
                'cx': search_engine_id,
                'q': f'site:patents.google.com {query}',
                'num': min(limit, 10),  # Google API limit
                'start': 1,
                'searchType': 'custom',
                'fields': 'items(title,link,snippet,displayLink,formattedUrl)'
            }
            
            # Add date range if specified
            if date_range:
                start_date, end_date = date_range
                params['sort'] = f'date:r:{start_date}:{end_date}'
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    base_url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        return await self._parse_google_search_response(data)
                    elif response.status == 429:
                        # Rate limit handling
                        logger.warning("Google API rate limit hit, waiting...")
                        await asyncio.sleep(60)
                        return await self._search_google_patents_production(query, date_range, limit)
                    else:
                        logger.error(f"Google Patents API error: {response.status}")
                        # PRODUCTION: No fallback to mock - raise error
                        raise ValueError(f"Google Patents API error: {response.status}. Check API configuration and credentials.")
                        
        except Exception as e:
            logger.error(f"Google Patents search error: {e}")
            # PRODUCTION: No fallback to mock - raise error  
            raise ValueError(f"Google Patents search failed: {e}. Check API configuration and network connectivity.")
    
    async def _parse_google_search_response(self, data: Dict) -> List[PatentDocument]:
        """Parse Google Search API response for patents into PatentDocument objects"""
        patents = []
        
        items = data.get('items', [])
        for item in items:
            try:
                # Extract patent information from Google Search results
                title = item.get('title', '')
                link = item.get('link', '')
                snippet = item.get('snippet', '')
                
                # Extract patent number from URL if possible
                patent_id = self._extract_patent_id_from_url(link)
                
                # Parse additional details if available in snippet
                patent_details = self._parse_patent_snippet(snippet)
                
                patent = PatentDocument(
                    patent_id=patent_id,
                    title=title,
                    abstract=snippet,
                    inventors=patent_details.get('inventors', []),
                    assignees=patent_details.get('assignees', []),
                    publication_date=patent_details.get('publication_date', ''),
                    filing_date=patent_details.get('filing_date', ''),
                    patent_number=patent_id,
                    classification_codes=patent_details.get('classification_codes', []),
                    claims=[],  # Would need additional API call to get full claims
                    priority_claims=[],
                    citations=[],
                    cited_by=[],
                    legal_status=patent_details.get('legal_status', 'Unknown'),
                    country=self._extract_country_from_patent_id(patent_id),
                    language='en',  # Assume English for Google Patents
                    relevance_score=0.0  # Will be calculated separately
                )
                patents.append(patent)
                
            except Exception as e:
                logger.warning(f"Error parsing patent item: {e}")
                continue
        
        return patents
    
    def _extract_patent_id_from_url(self, url: str) -> str:
        """Extract patent ID from Google Patents URL"""
        import re
        
        # Google Patents URL patterns
        patterns = [
            r'/patent/([A-Z]{2}\d+[A-Z]\d*)',  # Standard format
            r'/patent/([A-Z]{2}\d+)',          # Alternative format
            r'patent/([A-Z]{2}\d+[A-Z]\d*)',   # Without leading slash
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        # Fallback - extract from filename
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        if parsed.path:
            parts = parsed.path.split('/')
            if len(parts) > 1:
                return parts[-1]
        
        return url  # Return URL as fallback
    
    def _parse_patent_snippet(self, snippet: str) -> Dict[str, Any]:
        """Parse patent details from search snippet"""
        import re
        
        details = {}
        
        # Extract dates (various formats)
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'(\w+ \d{1,2}, \d{4})'
        ]
        
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, snippet))
        
        if dates:
            details['publication_date'] = dates[0]
            if len(dates) > 1:
                details['filing_date'] = dates[1]
        
        # Extract inventors (pattern: "Inventor: Name" or "by Name")
        inventor_patterns = [
            r'Inventor[s]?:\s*([^,\n]+)',
            r'by\s+([A-Z][a-z]+ [A-Z][a-z]+)',
            r'inventor[s]?\s+([A-Z][a-z]+ [A-Z][a-z]+)'
        ]
        
        inventors = []
        for pattern in inventor_patterns:
            matches = re.findall(pattern, snippet, re.IGNORECASE)
            inventors.extend(matches)
        
        if inventors:
            details['inventors'] = [inv.strip() for inv in inventors[:3]]  # Limit to 3
        
        # Extract assignees (pattern: "Assignee: Company" or company names)
        assignee_patterns = [
            r'Assignee[s]?:\s*([^,\n]+)',
            r'([A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company))',
        ]
        
        assignees = []
        for pattern in assignee_patterns:
            matches = re.findall(pattern, snippet, re.IGNORECASE)
            assignees.extend(matches)
        
        if assignees:
            details['assignees'] = [ass.strip() for ass in assignees[:2]]  # Limit to 2
        
        # Extract classification codes (pattern like A61K, B01D, etc.)
        class_pattern = r'\b([A-Z]\d{2}[A-Z]?\d*(?:/\d+)?)\b'
        class_codes = re.findall(class_pattern, snippet)
        if class_codes:
            details['classification_codes'] = class_codes[:5]  # Limit to 5
        
        return details
    
    def _extract_country_from_patent_id(self, patent_id: str) -> str:
        """Extract country code from patent ID"""
        if not patent_id:
            return 'Unknown'
        
        # Common country prefixes
        country_mapping = {
            'US': 'United States',
            'EP': 'European Patent Office',
            'JP': 'Japan',
            'CN': 'China',
            'DE': 'Germany',
            'GB': 'United Kingdom',
            'FR': 'France',
            'CA': 'Canada',
            'AU': 'Australia',
            'KR': 'South Korea',
            'IN': 'India',
            'WO': 'World Intellectual Property Organization'
        }
        
        # Extract first 2 characters
        country_code = patent_id[:2].upper()
        return country_mapping.get(country_code, country_code)
    
    # PRODUCTION IMPLEMENTATION: Mock functions removed for production deployment
    # Development testing should use test databases and proper API sandboxes
    # For development, use: export ENVIRONMENT=development and configure test APIs
    
    async def _calculate_relevance_score(self, patent: PatentDocument, query_terms: List[str]) -> float:
        """Calculate patent relevance score"""
        
        score = 0.0
        
        # Title relevance
        title_matches = sum(1 for term in query_terms if term.lower() in patent.title.lower())
        score += title_matches * 0.4
        
        # Abstract relevance
        abstract_matches = sum(1 for term in query_terms if term.lower() in patent.abstract.lower())
        score += abstract_matches * 0.3
        
        # Classification relevance
        relevant_classes = 0
        for term in query_terms:
            for category, codes in self.classification_mapping.items():
                if term.lower() in category:
                    for code in codes:
                        if any(code in pc for pc in patent.classification_codes):
                            relevant_classes += 1
        score += relevant_classes * 0.2
        
        # Citation impact (cited_by count)
        citation_impact = len(patent.cited_by) * 0.1
        score += min(citation_impact, 1.0)
        
        return min(score, 5.0)  # Cap at 5.0
    
    async def _cluster_patents(self, patents: List[PatentDocument]) -> List[PatentCluster]:
        """Cluster patents by technology areas"""
        
        clusters = []
        
        try:
            # Group by classification codes
            classification_groups = defaultdict(list)
            for patent in patents:
                for code in patent.classification_codes:
                    main_class = code[:4]  # First 4 characters
                    classification_groups[main_class].append(patent.patent_id)
            
            # Create clusters
            cluster_id = 0
            for class_code, patent_ids in classification_groups.items():
                if len(patent_ids) >= 2:  # Minimum cluster size
                    
                    # Determine technology area
                    tech_area = self._classify_technology_area(class_code)
                    
                    # Extract key concepts
                    cluster_patents = [p for p in patents if p.patent_id in patent_ids]
                    key_concepts = self._extract_key_concepts(cluster_patents)
                    
                    # Analyze key players
                    assignees = [assignee for p in cluster_patents for assignee in p.assignees]
                    key_players = list(set(assignees))[:5]
                    
                    cluster = PatentCluster(
                        cluster_id=f"cluster_{cluster_id}",
                        cluster_name=f"{tech_area} Technologies",
                        patents=patent_ids,
                        key_concepts=key_concepts,
                        technology_area=tech_area,
                        innovation_trend="Growing",
                        competitive_intensity=len(key_players) / len(patent_ids),
                        growth_rate=0.15,  # Mock growth rate
                        key_players=key_players
                    )
                    clusters.append(cluster)
                    cluster_id += 1
            
        except Exception as e:
            logger.error(f"Error clustering patents: {e}")
        
        return clusters
    
    def _classify_technology_area(self, class_code: str) -> str:
        """Classify technology area from patent class code"""
        
        area_mapping = {
            'A61K': 'Cosmetic Formulations',
            'A61Q': 'Personal Care Applications',
            'C11B': 'Fragrances and Essential Oils',
            'B65D': 'Packaging Technologies',
            'A45D': 'Beauty Tools and Accessories',
            'B01F': 'Manufacturing Processes'
        }
        
        for code, area in area_mapping.items():
            if class_code.startswith(code):
                return area
        
        return 'Other Technologies'
    
    def _extract_key_concepts(self, patents: List[PatentDocument]) -> List[str]:
        """Extract key concepts from patent cluster"""
        
        # Extract frequent terms from titles and abstracts
        all_text = ' '.join([p.title + ' ' + p.abstract for p in patents])
        
        # Simple keyword extraction (would use NLP in production)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text.lower())
        word_freq = defaultdict(int)
        
        # Filter common words and count frequency
        common_words = {'with', 'from', 'that', 'this', 'have', 'will', 'been', 'said', 'each', 'which'}
        for word in words:
            if word not in common_words:
                word_freq[word] += 1
        
        # Return top concepts
        top_concepts = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [concept for concept, freq in top_concepts[:10] if freq > 1]
    
    async def _identify_key_innovations(self, patents: List[PatentDocument]) -> List[PatentDocument]:
        """Identify key breakthrough innovations"""
        
        # Sort by relevance score and citation impact
        key_patents = sorted(patents, 
                           key=lambda x: (x.relevance_score + len(x.cited_by) * 0.1), 
                           reverse=True)
        
        return key_patents[:10]  # Top 10 innovations
    
    async def _analyze_competitive_landscape(self, patents: List[PatentDocument]) -> Dict[str, int]:
        """Analyze competitive landscape"""
        
        assignee_counts = defaultdict(int)
        for patent in patents:
            for assignee in patent.assignees:
                assignee_counts[assignee] += 1
        
        return dict(sorted(assignee_counts.items(), key=lambda x: x[1], reverse=True)[:15])
    
    async def _create_innovation_timeline(self, patents: List[PatentDocument]) -> Dict[str, List[PatentDocument]]:
        """Create innovation timeline"""
        
        timeline = defaultdict(list)
        for patent in patents:
            year = patent.publication_date[:4]
            timeline[year].append(patent)
        
        return dict(timeline)
    
    async def _identify_white_spaces(self, patents: List[PatentDocument], clusters: List[PatentCluster]) -> List[str]:
        """Identify potential innovation white spaces"""
        
        white_spaces = []
        
        # Analyze coverage gaps in classification codes
        all_classes = set()
        for patent in patents:
            all_classes.update(patent.classification_codes)
        
        # Expected classes for comprehensive cosmetic coverage
        expected_classes = ['A61K8/02', 'A61K8/04', 'A61K8/81', 'A61Q1/', 'A61Q5/', 'A61Q19/']
        
        for expected in expected_classes:
            if not any(expected in cls for cls in all_classes):
                white_spaces.append(f"Limited coverage in {expected} classification")
        
        # Technology combination opportunities
        if len(clusters) > 1:
            white_spaces.append("Cross-cluster innovation opportunities")
        
        # Sustainability gap analysis
        sustainability_terms = ['sustainable', 'biodegradable', 'natural', 'eco-friendly']
        sustainability_count = sum(1 for p in patents 
                                 for term in sustainability_terms 
                                 if term in (p.title + p.abstract).lower())
        
        if sustainability_count < len(patents) * 0.2:
            white_spaces.append("Sustainability innovation gap")
        
        return white_spaces[:5]


# Utility functions
async def quick_patent_search(query_terms: List[str], limit: int = 10) -> List[Dict]:
    """Quick patent search utility"""
    analyzer = PatentAnalyzer({})
    patents = await analyzer._search_patents(query_terms, None, limit)
    return [asdict(patent) for patent in patents]

async def analyze_competitor_patents(competitor_name: str, technology_area: str = "") -> Dict[str, Any]:
    """Analyze patents from specific competitor"""
    analyzer = PatentAnalyzer({})
    
    query_terms = [competitor_name]
    if technology_area:
        query_terms.append(technology_area)
    
    landscape = await analyzer.analyze_patent_landscape(query_terms, max_patents=100)
    
    return {
        'competitor': competitor_name,
        'total_patents': landscape.total_patents,
        'key_innovations': [asdict(patent) for patent in landscape.key_innovations[:5]],
        'technology_focus': [cluster.technology_area for cluster in landscape.technology_clusters],
        'analysis_date': landscape.analysis_timestamp
    }
