"""
Academic Publishing Simulation Environment
Comprehensive simulation framework for testing autonomous agents in academic publishing workflows
"""

import random
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    RESEARCH_DISCOVERY = "research_discovery"
    SUBMISSION_ASSISTANT = "submission_assistant"
    EDITORIAL_ORCHESTRATION = "editorial_orchestration"
    REVIEW_COORDINATION = "review_coordination"
    CONTENT_QUALITY = "content_quality"
    PUBLISHING_PRODUCTION = "publishing_production"
    ANALYTICS_MONITORING = "analytics_monitoring"

class ManuscriptStatus(Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    REVISION_REQUESTED = "revision_requested"
    REVISED = "revised"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PUBLISHED = "published"
    WITHDRAWN = "withdrawn"

class ReviewDecision(Enum):
    ACCEPT = "accept"
    MINOR_REVISION = "minor_revision"
    MAJOR_REVISION = "major_revision"
    REJECT = "reject"

class Domain(Enum):
    COMPUTER_SCIENCE = "computer_science"
    MEDICINE = "medicine"
    PHYSICS = "physics"
    BIOLOGY = "biology"
    CHEMISTRY = "chemistry"

@dataclass
class Author:
    id: str
    name: str
    affiliation: str
    expertise_areas: List[str]
    career_stage: str  # "student", "postdoc", "assistant", "associate", "full"
    h_index: int
    publication_count: int
    collaboration_network: List[str]

@dataclass
class Venue:
    id: str
    name: str
    type: str  # "journal", "conference"
    domain: Domain
    impact_factor: float
    acceptance_rate: float
    review_time_avg: int  # days
    prestige_score: float
    open_access: bool

@dataclass
class Manuscript:
    id: str
    title: str
    abstract: str
    authors: List[str]  # Author IDs
    domain: Domain
    keywords: List[str]
    submission_date: datetime
    status: ManuscriptStatus
    venue_id: Optional[str]
    quality_score: float  # 0-10
    novelty_score: float  # 0-10
    clarity_score: float  # 0-10
    significance_score: float  # 0-10
    review_history: List[Dict]
    revision_count: int
    citation_potential: float

@dataclass
class Review:
    id: str
    manuscript_id: str
    reviewer_id: str
    decision: ReviewDecision
    confidence: float  # 1-5
    quality_score: float  # 1-10
    novelty_score: float  # 1-10
    clarity_score: float  # 1-10
    significance_score: float  # 1-10
    comments: str
    review_time: int  # days taken
    submitted_date: datetime

@dataclass
class SimulationMetrics:
    total_manuscripts: int
    accepted_manuscripts: int
    rejected_manuscripts: int
    avg_review_time: float
    avg_revision_cycles: float
    agent_efficiency_scores: Dict[str, float]
    quality_improvement_rate: float
    collaboration_success_rate: float
    venue_match_accuracy: float
    overall_satisfaction: float

class AcademicPublishingSimulator:
    """
    Comprehensive simulation environment for academic publishing workflows
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.current_time = datetime.now()
        self.simulation_speed = config.get('simulation_speed', 1.0)  # 1.0 = real time
        
        # Initialize data structures
        self.authors: Dict[str, Author] = {}
        self.venues: Dict[str, Venue] = {}
        self.manuscripts: Dict[str, Manuscript] = {}
        self.reviews: Dict[str, Review] = {}
        self.agents: Dict[str, Any] = {}
        
        # Simulation state
        self.event_queue = deque()
        self.metrics = SimulationMetrics(0, 0, 0, 0, 0, {}, 0, 0, 0, 0)
        self.running = False
        
        # Initialize simulation environment
        self._initialize_authors()
        self._initialize_venues()
        self._initialize_agents()
        
        logger.info("Academic Publishing Simulator initialized")

    def _initialize_authors(self):
        """Initialize a diverse set of authors with realistic profiles"""
        author_profiles = [
            # Computer Science authors
            {"name": "Dr. Alice Chen", "domain": Domain.COMPUTER_SCIENCE, "career_stage": "full", "h_index": 45},
            {"name": "Prof. Bob Martinez", "domain": Domain.COMPUTER_SCIENCE, "career_stage": "associate", "h_index": 28},
            {"name": "Sarah Kim", "domain": Domain.COMPUTER_SCIENCE, "career_stage": "student", "h_index": 3},
            
            # Medicine authors
            {"name": "Dr. Emily Johnson", "domain": Domain.MEDICINE, "career_stage": "full", "h_index": 52},
            {"name": "Dr. Michael Brown", "domain": Domain.MEDICINE, "career_stage": "assistant", "h_index": 15},
            {"name": "Lisa Wang", "domain": Domain.MEDICINE, "career_stage": "postdoc", "h_index": 8},
            
            # Physics authors
            {"name": "Prof. David Wilson", "domain": Domain.PHYSICS, "career_stage": "full", "h_index": 38},
            {"name": "Dr. Maria Garcia", "domain": Domain.PHYSICS, "career_stage": "associate", "h_index": 22},
            {"name": "James Lee", "domain": Domain.PHYSICS, "career_stage": "student", "h_index": 2},
        ]
        
        for i, profile in enumerate(author_profiles):
            author_id = f"author_{i+1}"
            
            # Generate expertise areas based on domain
            expertise_map = {
                Domain.COMPUTER_SCIENCE: ["machine learning", "algorithms", "systems", "HCI", "security"],
                Domain.MEDICINE: ["cardiology", "oncology", "neurology", "epidemiology", "surgery"],
                Domain.PHYSICS: ["quantum mechanics", "condensed matter", "astrophysics", "particle physics", "optics"]
            }
            
            expertise_areas = random.sample(expertise_map[profile["domain"]], k=random.randint(2, 4))
            
            author = Author(
                id=author_id,
                name=profile["name"],
                affiliation=f"University {chr(65 + i)}",
                expertise_areas=expertise_areas,
                career_stage=profile["career_stage"],
                h_index=profile["h_index"],
                publication_count=profile["h_index"] * random.randint(2, 4),
                collaboration_network=[]
            )
            
            self.authors[author_id] = author

    def _initialize_venues(self):
        """Initialize academic venues with realistic characteristics"""
        venue_data = [
            # Computer Science venues
            {"name": "Nature Machine Intelligence", "type": "journal", "domain": Domain.COMPUTER_SCIENCE, 
             "impact_factor": 25.8, "acceptance_rate": 0.15, "review_time": 120, "prestige": 0.95},
            {"name": "NeurIPS", "type": "conference", "domain": Domain.COMPUTER_SCIENCE,
             "impact_factor": 12.0, "acceptance_rate": 0.25, "review_time": 90, "prestige": 0.90},
            {"name": "IEEE Transactions on Computers", "type": "journal", "domain": Domain.COMPUTER_SCIENCE,
             "impact_factor": 3.2, "acceptance_rate": 0.35, "review_time": 150, "prestige": 0.70},
            
            # Medicine venues
            {"name": "New England Journal of Medicine", "type": "journal", "domain": Domain.MEDICINE,
             "impact_factor": 91.2, "acceptance_rate": 0.05, "review_time": 180, "prestige": 1.0},
            {"name": "The Lancet", "type": "journal", "domain": Domain.MEDICINE,
             "impact_factor": 79.3, "acceptance_rate": 0.08, "review_time": 160, "prestige": 0.98},
            {"name": "JAMA", "type": "journal", "domain": Domain.MEDICINE,
             "impact_factor": 56.3, "acceptance_rate": 0.12, "review_time": 140, "prestige": 0.95},
            
            # Physics venues
            {"name": "Physical Review Letters", "type": "journal", "domain": Domain.PHYSICS,
             "impact_factor": 9.2, "acceptance_rate": 0.30, "review_time": 100, "prestige": 0.92},
            {"name": "Nature Physics", "type": "journal", "domain": Domain.PHYSICS,
             "impact_factor": 19.6, "acceptance_rate": 0.18, "review_time": 120, "prestige": 0.96},
            {"name": "Physical Review A", "type": "journal", "domain": Domain.PHYSICS,
             "impact_factor": 3.1, "acceptance_rate": 0.45, "review_time": 80, "prestige": 0.75},
        ]
        
        for i, venue_info in enumerate(venue_data):
            venue_id = f"venue_{i+1}"
            venue = Venue(
                id=venue_id,
                name=venue_info["name"],
                type=venue_info["type"],
                domain=venue_info["domain"],
                impact_factor=venue_info["impact_factor"],
                acceptance_rate=venue_info["acceptance_rate"],
                review_time_avg=venue_info["review_time"],
                prestige_score=venue_info["prestige"],
                open_access=random.choice([True, False])
            )
            self.venues[venue_id] = venue

    def _initialize_agents(self):
        """Initialize autonomous agents for the simulation"""
        agent_configs = [
            {"type": AgentType.RESEARCH_DISCOVERY, "efficiency": 0.85, "accuracy": 0.80},
            {"type": AgentType.SUBMISSION_ASSISTANT, "efficiency": 0.90, "accuracy": 0.85},
            {"type": AgentType.EDITORIAL_ORCHESTRATION, "efficiency": 0.75, "accuracy": 0.90},
            {"type": AgentType.REVIEW_COORDINATION, "efficiency": 0.80, "accuracy": 0.85},
            {"type": AgentType.CONTENT_QUALITY, "efficiency": 0.70, "accuracy": 0.95},
            {"type": AgentType.PUBLISHING_PRODUCTION, "efficiency": 0.95, "accuracy": 0.90},
            {"type": AgentType.ANALYTICS_MONITORING, "efficiency": 0.90, "accuracy": 0.85},
        ]
        
        for config in agent_configs:
            agent_id = f"agent_{config['type'].value}"
            self.agents[agent_id] = {
                "type": config["type"],
                "efficiency": config["efficiency"],
                "accuracy": config["accuracy"],
                "workload": 0,
                "performance_history": [],
                "active": True
            }

    def generate_manuscript(self, author_ids: List[str], domain: Domain) -> Manuscript:
        """Generate a realistic manuscript with quality scores"""
        manuscript_id = str(uuid.uuid4())
        
        # Generate quality scores based on author expertise
        lead_author = self.authors[author_ids[0]]
        base_quality = min(10, max(1, lead_author.h_index / 10 + random.gauss(0, 1)))
        
        # Add collaboration bonus
        collaboration_bonus = min(2, len(author_ids) * 0.3)
        
        manuscript = Manuscript(
            id=manuscript_id,
            title=f"Research Paper {manuscript_id[:8]}",
            abstract=f"Abstract for manuscript {manuscript_id[:8]}",
            authors=author_ids,
            domain=domain,
            keywords=random.sample(["AI", "ML", "systems", "theory", "applications"], k=3),
            submission_date=self.current_time,
            status=ManuscriptStatus.DRAFT,
            venue_id=None,
            quality_score=min(10, base_quality + collaboration_bonus + random.gauss(0, 0.5)),
            novelty_score=random.uniform(3, 9),
            clarity_score=random.uniform(4, 8),
            significance_score=random.uniform(3, 8),
            review_history=[],
            revision_count=0,
            citation_potential=random.uniform(0, 100)
        )
        
        self.manuscripts[manuscript_id] = manuscript
        return manuscript

    def simulate_agent_action(self, agent_type: AgentType, action: str, context: Dict) -> Dict:
        """Simulate an agent performing an action"""
        agent_id = f"agent_{agent_type.value}"
        agent = self.agents[agent_id]
        
        # Simulate processing time based on agent efficiency
        processing_time = random.uniform(1, 10) / agent["efficiency"]
        
        # Simulate action success based on agent accuracy
        success = random.random() < agent["accuracy"]
        
        result = {
            "agent_id": agent_id,
            "action": action,
            "success": success,
            "processing_time": processing_time,
            "timestamp": self.current_time,
            "context": context
        }
        
        # Update agent performance
        agent["workload"] += 1
        agent["performance_history"].append({
            "action": action,
            "success": success,
            "time": processing_time,
            "timestamp": self.current_time
        })
        
        return result

    def simulate_manuscript_submission(self, manuscript_id: str, venue_id: str) -> Dict:
        """Simulate manuscript submission process"""
        manuscript = self.manuscripts[manuscript_id]
        venue = self.venues[venue_id]
        
        # Submission Assistant Agent helps with submission
        submission_result = self.simulate_agent_action(
            AgentType.SUBMISSION_ASSISTANT,
            "prepare_submission",
            {"manuscript_id": manuscript_id, "venue_id": venue_id}
        )
        
        if submission_result["success"]:
            manuscript.status = ManuscriptStatus.SUBMITTED
            manuscript.venue_id = venue_id
            
            # Schedule initial review
            review_start_date = self.current_time + timedelta(days=random.randint(1, 7))
            self.event_queue.append({
                "type": "start_review",
                "manuscript_id": manuscript_id,
                "scheduled_time": review_start_date
            })
            
            logger.info(f"Manuscript {manuscript_id} submitted to {venue.name}")
        
        return submission_result

    def simulate_peer_review(self, manuscript_id: str) -> List[Review]:
        """Simulate peer review process"""
        manuscript = self.manuscripts[manuscript_id]
        venue = self.venues[manuscript.venue_id]
        
        # Review Coordination Agent assigns reviewers
        coordination_result = self.simulate_agent_action(
            AgentType.REVIEW_COORDINATION,
            "assign_reviewers",
            {"manuscript_id": manuscript_id}
        )
        
        reviews = []
        num_reviewers = random.randint(2, 4)
        
        for i in range(num_reviewers):
            # Simulate reviewer selection and review process
            reviewer_id = f"reviewer_{random.randint(1, 100)}"
            
            # Generate review based on manuscript quality and venue standards
            base_score = manuscript.quality_score
            venue_bias = venue.prestige_score * 2  # Higher prestige venues are more selective
            
            review_scores = {
                "quality": max(1, min(10, base_score + random.gauss(0, 1))),
                "novelty": max(1, min(10, manuscript.novelty_score + random.gauss(0, 1))),
                "clarity": max(1, min(10, manuscript.clarity_score + random.gauss(0, 1))),
                "significance": max(1, min(10, manuscript.significance_score + random.gauss(0, 1)))
            }
            
            avg_score = sum(review_scores.values()) / len(review_scores)
            
            # Determine decision based on scores and venue acceptance rate
            if avg_score >= 8 and random.random() < venue.acceptance_rate * 2:
                decision = ReviewDecision.ACCEPT
            elif avg_score >= 6:
                decision = random.choice([ReviewDecision.MINOR_REVISION, ReviewDecision.MAJOR_REVISION])
            else:
                decision = ReviewDecision.REJECT
            
            review = Review(
                id=str(uuid.uuid4()),
                manuscript_id=manuscript_id,
                reviewer_id=reviewer_id,
                decision=decision,
                confidence=random.uniform(3, 5),
                quality_score=review_scores["quality"],
                novelty_score=review_scores["novelty"],
                clarity_score=review_scores["clarity"],
                significance_score=review_scores["significance"],
                comments=f"Review comments for manuscript {manuscript_id[:8]}",
                review_time=random.randint(14, venue.review_time_avg),
                submitted_date=self.current_time + timedelta(days=random.randint(14, 60))
            )
            
            reviews.append(review)
            self.reviews[review.id] = review
        
        # Editorial Orchestration Agent makes final decision
        editorial_result = self.simulate_agent_action(
            AgentType.EDITORIAL_ORCHESTRATION,
            "make_editorial_decision",
            {"manuscript_id": manuscript_id, "reviews": [r.id for r in reviews]}
        )
        
        # Determine final decision based on reviews
        decisions = [r.decision for r in reviews]
        if ReviewDecision.ACCEPT in decisions and len([d for d in decisions if d == ReviewDecision.ACCEPT]) >= len(decisions) / 2:
            manuscript.status = ManuscriptStatus.ACCEPTED
        elif ReviewDecision.REJECT in decisions and len([d for d in decisions if d == ReviewDecision.REJECT]) >= len(decisions) / 2:
            manuscript.status = ManuscriptStatus.REJECTED
        else:
            manuscript.status = ManuscriptStatus.REVISION_REQUESTED
            manuscript.revision_count += 1
        
        manuscript.review_history.extend([asdict(r) for r in reviews])
        
        logger.info(f"Manuscript {manuscript_id} review completed: {manuscript.status.value}")
        return reviews

    def simulate_revision_process(self, manuscript_id: str) -> bool:
        """Simulate manuscript revision process"""
        manuscript = self.manuscripts[manuscript_id]
        
        # Content Quality Agent helps with revisions
        quality_result = self.simulate_agent_action(
            AgentType.CONTENT_QUALITY,
            "assist_revision",
            {"manuscript_id": manuscript_id}
        )
        
        if quality_result["success"]:
            # Improve manuscript scores based on revision
            improvement_factor = random.uniform(0.1, 0.3)
            manuscript.quality_score = min(10, manuscript.quality_score * (1 + improvement_factor))
            manuscript.clarity_score = min(10, manuscript.clarity_score * (1 + improvement_factor))
            manuscript.status = ManuscriptStatus.REVISED
            
            # Schedule re-review
            re_review_date = self.current_time + timedelta(days=random.randint(7, 21))
            self.event_queue.append({
                "type": "start_review",
                "manuscript_id": manuscript_id,
                "scheduled_time": re_review_date
            })
            
            return True
        
        return False

    def simulate_publication_process(self, manuscript_id: str) -> Dict:
        """Simulate publication production process"""
        manuscript = self.manuscripts[manuscript_id]
        
        # Publishing Production Agent handles publication
        publication_result = self.simulate_agent_action(
            AgentType.PUBLISHING_PRODUCTION,
            "prepare_publication",
            {"manuscript_id": manuscript_id}
        )
        
        if publication_result["success"]:
            manuscript.status = ManuscriptStatus.PUBLISHED
            
            # Simulate citation accumulation
            venue = self.venues[manuscript.venue_id]
            citation_multiplier = venue.impact_factor / 10
            expected_citations = manuscript.citation_potential * citation_multiplier
            
            publication_result["expected_citations"] = expected_citations
            publication_result["publication_date"] = self.current_time
            
            logger.info(f"Manuscript {manuscript_id} published successfully")
        
        return publication_result

    def run_simulation_step(self):
        """Execute one simulation step"""
        # Process scheduled events
        current_events = []
        while self.event_queue and self.event_queue[0]["scheduled_time"] <= self.current_time:
            current_events.append(self.event_queue.popleft())
        
        for event in current_events:
            if event["type"] == "start_review":
                self.simulate_peer_review(event["manuscript_id"])
            elif event["type"] == "revision_deadline":
                manuscript = self.manuscripts[event["manuscript_id"]]
                if manuscript.status == ManuscriptStatus.REVISION_REQUESTED:
                    self.simulate_revision_process(event["manuscript_id"])
        
        # Generate random events (new submissions, etc.)
        if random.random() < 0.1:  # 10% chance of new submission per step
            # Select random authors and domain
            author_ids = random.sample(list(self.authors.keys()), k=random.randint(1, 3))
            domain = random.choice(list(Domain))
            
            # Generate and submit manuscript
            manuscript = self.generate_manuscript(author_ids, domain)
            
            # Select appropriate venue
            domain_venues = [v for v in self.venues.values() if v.domain == domain]
            if domain_venues:
                venue = random.choice(domain_venues)
                self.simulate_manuscript_submission(manuscript.id, venue.id)
        
        # Update simulation time
        self.current_time += timedelta(hours=1 * self.simulation_speed)

    def run_simulation(self, duration_days: int = 30):
        """Run the simulation for specified duration"""
        self.running = True
        end_time = self.current_time + timedelta(days=duration_days)
        
        logger.info(f"Starting simulation for {duration_days} days")
        
        step_count = 0
        while self.running and self.current_time < end_time:
            self.run_simulation_step()
            step_count += 1
            
            # Log progress every 24 steps (1 day)
            if step_count % 24 == 0:
                self.update_metrics()
                logger.info(f"Simulation day {step_count // 24}: {self.metrics.total_manuscripts} manuscripts processed")
        
        self.running = False
        self.update_metrics()
        logger.info("Simulation completed")

    def update_metrics(self):
        """Update simulation metrics"""
        total_manuscripts = len(self.manuscripts)
        accepted = len([m for m in self.manuscripts.values() if m.status == ManuscriptStatus.ACCEPTED])
        rejected = len([m for m in self.manuscripts.values() if m.status == ManuscriptStatus.REJECTED])
        published = len([m for m in self.manuscripts.values() if m.status == ManuscriptStatus.PUBLISHED])
        
        # Calculate average review time
        completed_reviews = [r for r in self.reviews.values() if r.submitted_date <= self.current_time]
        avg_review_time = np.mean([r.review_time for r in completed_reviews]) if completed_reviews else 0
        
        # Calculate agent efficiency scores
        agent_scores = {}
        for agent_id, agent in self.agents.items():
            if agent["performance_history"]:
                success_rate = np.mean([p["success"] for p in agent["performance_history"]])
                avg_time = np.mean([p["time"] for p in agent["performance_history"]])
                efficiency_score = success_rate / (avg_time / 10)  # Normalize by expected time
                agent_scores[agent_id] = efficiency_score
        
        self.metrics = SimulationMetrics(
            total_manuscripts=total_manuscripts,
            accepted_manuscripts=accepted + published,
            rejected_manuscripts=rejected,
            avg_review_time=avg_review_time,
            avg_revision_cycles=np.mean([m.revision_count for m in self.manuscripts.values()]) if self.manuscripts else 0,
            agent_efficiency_scores=agent_scores,
            quality_improvement_rate=0.15,  # Placeholder
            collaboration_success_rate=0.75,  # Placeholder
            venue_match_accuracy=0.80,  # Placeholder
            overall_satisfaction=0.78  # Placeholder
        )

    def get_simulation_report(self) -> Dict:
        """Generate comprehensive simulation report"""
        self.update_metrics()
        
        # Manuscript status distribution
        status_distribution = defaultdict(int)
        for manuscript in self.manuscripts.values():
            status_distribution[manuscript.status.value] += 1
        
        # Domain distribution
        domain_distribution = defaultdict(int)
        for manuscript in self.manuscripts.values():
            domain_distribution[manuscript.domain.value] += 1
        
        # Agent performance summary
        agent_performance = {}
        for agent_id, agent in self.agents.items():
            if agent["performance_history"]:
                performance = {
                    "total_actions": len(agent["performance_history"]),
                    "success_rate": np.mean([p["success"] for p in agent["performance_history"]]),
                    "avg_processing_time": np.mean([p["time"] for p in agent["performance_history"]]),
                    "efficiency_score": self.metrics.agent_efficiency_scores.get(agent_id, 0)
                }
                agent_performance[agent_id] = performance
        
        # Venue performance
        venue_performance = {}
        for venue_id, venue in self.venues.items():
            venue_manuscripts = [m for m in self.manuscripts.values() if m.venue_id == venue_id]
            if venue_manuscripts:
                accepted_count = len([m for m in venue_manuscripts if m.status in [ManuscriptStatus.ACCEPTED, ManuscriptStatus.PUBLISHED]])
                venue_performance[venue_id] = {
                    "name": venue.name,
                    "submissions": len(venue_manuscripts),
                    "acceptances": accepted_count,
                    "acceptance_rate": accepted_count / len(venue_manuscripts) if venue_manuscripts else 0,
                    "avg_quality": np.mean([m.quality_score for m in venue_manuscripts])
                }
        
        report = {
            "simulation_summary": {
                "duration": str(self.current_time - datetime.now()),
                "total_manuscripts": self.metrics.total_manuscripts,
                "acceptance_rate": self.metrics.accepted_manuscripts / self.metrics.total_manuscripts if self.metrics.total_manuscripts > 0 else 0,
                "avg_review_time_days": self.metrics.avg_review_time,
                "avg_revision_cycles": self.metrics.avg_revision_cycles
            },
            "manuscript_distribution": dict(status_distribution),
            "domain_distribution": dict(domain_distribution),
            "agent_performance": agent_performance,
            "venue_performance": venue_performance,
            "quality_metrics": {
                "avg_manuscript_quality": np.mean([m.quality_score for m in self.manuscripts.values()]) if self.manuscripts else 0,
                "quality_improvement_rate": self.metrics.quality_improvement_rate,
                "collaboration_success_rate": self.metrics.collaboration_success_rate,
                "venue_match_accuracy": self.metrics.venue_match_accuracy
            },
            "system_performance": {
                "overall_satisfaction": self.metrics.overall_satisfaction,
                "agent_efficiency_scores": self.metrics.agent_efficiency_scores,
                "bottlenecks": self._identify_bottlenecks(),
                "recommendations": self._generate_recommendations()
            }
        }
        
        return report

    def _identify_bottlenecks(self) -> List[str]:
        """Identify system bottlenecks"""
        bottlenecks = []
        
        # Check for slow agents
        for agent_id, score in self.metrics.agent_efficiency_scores.items():
            if score < 0.5:
                bottlenecks.append(f"Low efficiency in {agent_id}")
        
        # Check for long review times
        if self.metrics.avg_review_time > 90:
            bottlenecks.append("Review process taking too long")
        
        # Check for high revision rates
        if self.metrics.avg_revision_cycles > 2:
            bottlenecks.append("High revision cycle count")
        
        return bottlenecks

    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if self.metrics.avg_review_time > 90:
            recommendations.append("Consider adding more reviewers or improving reviewer assignment")
        
        if self.metrics.quality_improvement_rate < 0.1:
            recommendations.append("Enhance content quality agent capabilities")
        
        if self.metrics.venue_match_accuracy < 0.7:
            recommendations.append("Improve venue recommendation algorithms")
        
        return recommendations

    def export_results(self, filename: str):
        """Export simulation results to JSON file"""
        report = self.get_simulation_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Simulation results exported to {filename}")

# Example usage and testing
if __name__ == "__main__":
    # Configuration for simulation
    config = {
        "simulation_speed": 24.0,  # 24x speed (1 hour = 1 minute)
        "random_seed": 42
    }
    
    # Set random seed for reproducibility
    random.seed(config.get("random_seed", 42))
    np.random.seed(config.get("random_seed", 42))
    
    # Initialize and run simulation
    simulator = AcademicPublishingSimulator(config)
    
    # Run simulation for 30 days
    simulator.run_simulation(duration_days=30)
    
    # Generate and display report
    report = simulator.get_simulation_report()
    print("\n" + "="*50)
    print("SIMULATION REPORT")
    print("="*50)
    
    print(f"\nTotal Manuscripts: {report['simulation_summary']['total_manuscripts']}")
    print(f"Acceptance Rate: {report['simulation_summary']['acceptance_rate']:.2%}")
    print(f"Average Review Time: {report['simulation_summary']['avg_review_time_days']:.1f} days")
    print(f"Average Revision Cycles: {report['simulation_summary']['avg_revision_cycles']:.1f}")
    
    print("\nAgent Performance:")
    for agent_id, performance in report['agent_performance'].items():
        print(f"  {agent_id}: {performance['success_rate']:.2%} success, {performance['efficiency_score']:.2f} efficiency")
    
    print("\nSystem Recommendations:")
    for rec in report['system_performance']['recommendations']:
        print(f"  - {rec}")
    
    # Export detailed results
    simulator.export_results("simulation_results.json")
    print(f"\nDetailed results exported to simulation_results.json")

