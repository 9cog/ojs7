"""
INCI Database Integration for Research Discovery Agent
Provides comprehensive cosmetic ingredient analysis and validation
"""
import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class INCIIngredient:
    """INCI ingredient data structure"""
    inci_name: str
    cas_number: Optional[str]
    function: List[str]
    safety_rating: Optional[str]
    restrictions: List[str]
    synonyms: List[str]
    description: str
    last_updated: str

@dataclass
class FormulationAnalysis:
    """Analysis results for a formulation"""
    ingredients: List[INCIIngredient]
    safety_score: float
    regulatory_compliance: Dict[str, bool]
    recommendations: List[str]
    warnings: List[str]
    analysis_timestamp: str

class INCIIntegrator:
    """Enhanced INCI database integration system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('inci_api_key')
        self.base_url = "https://api.incidecoder.com/v1"
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour cache
        
        # Regional regulatory databases
        self.regulatory_regions = {
            'EU': 'European Union',
            'US': 'United States', 
            'CA': 'Canada',
            'AU': 'Australia',
            'JP': 'Japan',
            'KR': 'South Korea'
        }
        
    async def search_ingredient(self, ingredient_name: str, region: str = 'EU') -> Optional[INCIIngredient]:
        """
        Search for a specific ingredient in INCI database
        Returns detailed ingredient information
        """
        cache_key = f"ingredient_{ingredient_name}_{region}"
        
        # Check cache first
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now().timestamp() - timestamp < self.cache_ttl:
                logger.debug(f"Cache hit for ingredient: {ingredient_name}")
                return INCIIngredient(**cached_data)
        
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    'ingredient': ingredient_name,
                    'region': region,
                    'format': 'json'
                }
                
                if self.api_key:
                    params['api_key'] = self.api_key
                
                async with session.get(f"{self.base_url}/ingredients/search", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('results'):
                            ingredient_data = data['results'][0]  # Take first match
                            
                            ingredient = INCIIngredient(
                                inci_name=ingredient_data.get('inci_name', ingredient_name),
                                cas_number=ingredient_data.get('cas_number'),
                                function=ingredient_data.get('function', []),
                                safety_rating=ingredient_data.get('safety_rating'),
                                restrictions=ingredient_data.get('restrictions', []),
                                synonyms=ingredient_data.get('synonyms', []),
                                description=ingredient_data.get('description', ''),
                                last_updated=datetime.now().isoformat()
                            )
                            
                            # Cache the result
                            self.cache[cache_key] = (asdict(ingredient), datetime.now().timestamp())
                            
                            logger.info(f"Successfully retrieved INCI data for: {ingredient_name}")
                            return ingredient
                        else:
                            logger.warning(f"No INCI data found for ingredient: {ingredient_name}")
                            return None
                    else:
                        logger.error(f"INCI API request failed with status {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error searching INCI ingredient {ingredient_name}: {e}")
            return None
    
    async def analyze_formulation(self, ingredients_list: List[str], region: str = 'EU') -> FormulationAnalysis:
        """
        Analyze a complete formulation for safety and regulatory compliance
        """
        logger.info(f"Analyzing formulation with {len(ingredients_list)} ingredients for region: {region}")
        
        analyzed_ingredients = []
        safety_scores = []
        all_restrictions = []
        recommendations = []
        warnings = []
        
        # Analyze each ingredient
        for ingredient_name in ingredients_list:
            ingredient_data = await self.search_ingredient(ingredient_name, region)
            
            if ingredient_data:
                analyzed_ingredients.append(ingredient_data)
                
                # Calculate safety score (simplified scoring system)
                if ingredient_data.safety_rating:
                    score = self._calculate_safety_score(ingredient_data.safety_rating)
                    safety_scores.append(score)
                
                # Collect restrictions
                if ingredient_data.restrictions:
                    all_restrictions.extend(ingredient_data.restrictions)
                
                # Generate recommendations
                if ingredient_data.function:
                    if 'preservative' in ingredient_data.function and len([i for i in analyzed_ingredients if 'preservative' in i.function]) > 3:
                        warnings.append(f"Multiple preservatives detected including {ingredient_data.inci_name}")
                
            else:
                warnings.append(f"Unknown ingredient: {ingredient_name} - requires manual review")
        
        # Calculate overall safety score
        overall_safety = sum(safety_scores) / len(safety_scores) if safety_scores else 0.5
        
        # Check regulatory compliance
        compliance = await self._check_regulatory_compliance(analyzed_ingredients, region)
        
        # Generate formulation recommendations
        recommendations.extend(self._generate_formulation_recommendations(analyzed_ingredients))
        
        return FormulationAnalysis(
            ingredients=analyzed_ingredients,
            safety_score=overall_safety,
            regulatory_compliance=compliance,
            recommendations=recommendations,
            warnings=warnings,
            analysis_timestamp=datetime.now().isoformat()
        )
    
    def _calculate_safety_score(self, safety_rating: str) -> float:
        """Calculate numeric safety score from rating"""
        rating_map = {
            'excellent': 1.0,
            'good': 0.8,
            'fair': 0.6,
            'poor': 0.4,
            'avoid': 0.2,
            'unknown': 0.5
        }
        return rating_map.get(safety_rating.lower(), 0.5)
    
    async def _check_regulatory_compliance(self, ingredients: List[INCIIngredient], region: str) -> Dict[str, bool]:
        """Check regulatory compliance for ingredients in specific region"""
        compliance = {}
        
        for ingredient in ingredients:
            # Simplified compliance check - in reality would query regulatory databases
            has_restrictions = bool(ingredient.restrictions)
            compliance[ingredient.inci_name] = not has_restrictions
            
        return compliance
    
    def _generate_formulation_recommendations(self, ingredients: List[INCIIngredient]) -> List[str]:
        """Generate intelligent formulation recommendations"""
        recommendations = []
        
        # Check for essential functions
        functions = [func for ingredient in ingredients for func in ingredient.function]
        
        if 'preservative' not in functions:
            recommendations.append("Consider adding a preservative system for product stability")
        
        if 'emulsifier' not in functions and any('oil' in ing.inci_name.lower() for ing in ingredients):
            recommendations.append("Emulsifier may be needed for oil-containing formulation")
        
        if 'antioxidant' not in functions:
            recommendations.append("Consider adding antioxidants to prevent oxidation")
        
        # Safety recommendations
        restricted_ingredients = [ing for ing in ingredients if ing.restrictions]
        if restricted_ingredients:
            recommendations.append(f"Review restrictions for: {', '.join([ing.inci_name for ing in restricted_ingredients])}")
        
        return recommendations
    
    async def batch_ingredient_lookup(self, ingredient_list: List[str], region: str = 'EU') -> Dict[str, INCIIngredient]:
        """
        Efficient batch lookup of multiple ingredients
        """
        logger.info(f"Performing batch lookup of {len(ingredient_list)} ingredients")
        
        tasks = [self.search_ingredient(ingredient, region) for ingredient in ingredient_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        batch_results = {}
        for i, (ingredient_name, result) in enumerate(zip(ingredient_list, results)):
            if isinstance(result, Exception):
                logger.error(f"Error processing ingredient {ingredient_name}: {result}")
                batch_results[ingredient_name] = None
            else:
                batch_results[ingredient_name] = result
        
        return batch_results
    
    async def get_ingredient_trends(self, category: str = 'skincare', limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get trending ingredients in cosmetic industry
        """
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    'category': category,
                    'limit': limit,
                    'sort': 'trending'
                }
                
                if self.api_key:
                    params['api_key'] = self.api_key
                
                async with session.get(f"{self.base_url}/trends", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('trending_ingredients', [])
                    else:
                        logger.error(f"Trends API request failed with status {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error fetching ingredient trends: {e}")
            return []
    
    def export_analysis_report(self, analysis: FormulationAnalysis, format: str = 'json') -> str:
        """
        Export formulation analysis as formatted report
        """
        if format.lower() == 'json':
            return json.dumps(asdict(analysis), indent=2, ensure_ascii=False)
        
        elif format.lower() == 'markdown':
            report = f"""# Formulation Analysis Report
            
## Summary
- **Analysis Date**: {analysis.analysis_timestamp}
- **Ingredients Analyzed**: {len(analysis.ingredients)}
- **Overall Safety Score**: {analysis.safety_score:.2f}/1.00
- **Regulatory Compliance**: {sum(analysis.regulatory_compliance.values())}/{len(analysis.regulatory_compliance)} ingredients compliant

## Ingredients Analysis
"""
            for ingredient in analysis.ingredients:
                report += f"""
### {ingredient.inci_name}
- **CAS Number**: {ingredient.cas_number or 'N/A'}
- **Function**: {', '.join(ingredient.function) if ingredient.function else 'N/A'}
- **Safety Rating**: {ingredient.safety_rating or 'Unknown'}
- **Restrictions**: {', '.join(ingredient.restrictions) if ingredient.restrictions else 'None'}
"""
            
            if analysis.recommendations:
                report += "\n## Recommendations\n"
                for rec in analysis.recommendations:
                    report += f"- {rec}\n"
            
            if analysis.warnings:
                report += "\n## Warnings\n"
                for warning in analysis.warnings:
                    report += f"- ⚠️ {warning}\n"
            
            return report
        
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Utility functions for quick INCI operations
async def quick_ingredient_check(ingredient_name: str, region: str = 'EU') -> bool:
    """Quick check if ingredient is recognized in INCI database"""
    integrator = INCIIntegrator({})
    result = await integrator.search_ingredient(ingredient_name, region)
    return result is not None

async def validate_ingredient_list(ingredients: List[str], region: str = 'EU') -> Tuple[List[str], List[str]]:
    """Validate list of ingredients, return (valid, invalid) lists"""
    integrator = INCIIntegrator({})
    results = await integrator.batch_ingredient_lookup(ingredients, region)
    
    valid = [name for name, data in results.items() if data is not None]
    invalid = [name for name, data in results.items() if data is None]
    
    return valid, invalid
