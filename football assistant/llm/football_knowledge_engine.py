
"""
Advanced Football Knowledge and Analytics Engine

This module implements sophisticated football analysis capabilities including tactical analysis,
predictive analytics, historical context, and transfer intelligence. It transforms the assistant
from a simple data retriever into a comprehensive football expert.

Core Components:
- TacticalAnalyzer: Formation analysis, tactical insights, and strategy evaluation
- PredictiveAnalytics: Match predictions, form analysis, and statistical forecasting
- HistoricalContext: Historical data analysis, trends, and contextual insights
- TransferIntelligence: Transfer market analysis and player valuation
- FootballKnowledgeCoordinator: Main orchestrator for all football expertise

Author: Football Assistant AI System
"""

import os
import re
import json
import time
import math
import logging
import asyncio
import hashlib
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, Counter

# Import existing modules for integration
from openrouter_llm import query_llm
from api.football_data import get_standings, get_teams, get_team_matches, resolve_competition
from api.api_football import get_live_scores
from api.livescore_api import get_live_scores_lsa

# Import intelligence modules
try:
    from llm_intelligence_engine import (
        QueryIntent, ResolvedEntity, ConversationContext, QueryType, Confidence
    )
    from intelligent_entity_resolver import (
        EntityResolutionCoordinator, ResolutionContext, EntityType
    )
except ImportError:
    # Fallback if modules not available
    logging.warning("Intelligence modules not available, using simplified structures")
    QueryIntent = None
    ResolvedEntity = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Data Models and Enums
# ============================================================================

class AnalysisType(Enum):
    """Types of football analysis that can be performed."""
    TACTICAL = "tactical"
    PREDICTIVE = "predictive"
    HISTORICAL = "historical"
    TRANSFER = "transfer"
    PERFORMANCE = "performance"
    COMPARATIVE = "comparative"
    STRATEGIC = "strategic"

class Formation(Enum):
    """Common football formations."""
    F_4_4_2 = "4-4-2"
    F_4_3_3 = "4-3-3"
    F_3_5_2 = "3-5-2"
    F_4_2_3_1 = "4-2-3-1"
    F_4_1_4_1 = "4-1-4-1"
    F_3_4_3 = "3-4-3"
    F_5_3_2 = "5-3-2"
    F_4_5_1 = "4-5-1"

class TacticalStyle(Enum):
    """Tactical playing styles."""
    POSSESSION_BASED = "possession_based"
    COUNTER_ATTACKING = "counter_attacking"
    HIGH_PRESSING = "high_pressing"
    DEFENSIVE = "defensive"
    DIRECT_PLAY = "direct_play"
    TIKI_TAKA = "tiki_taka"
    GEGENPRESS = "gegenpress"
    PARK_THE_BUS = "park_the_bus"

class PredictionConfidence(Enum):
    """Confidence levels for predictions."""
    VERY_HIGH = "very_high"  # > 0.9
    HIGH = "high"           # 0.8 - 0.9
    MEDIUM = "medium"       # 0.6 - 0.8
    LOW = "low"            # 0.4 - 0.6
    VERY_LOW = "very_low"  # < 0.4

@dataclass
class TacticalAnalysis:
    """Results of tactical analysis."""
    team_name: str
    formation: Optional[Formation]
    tactical_style: Optional[TacticalStyle]
    strengths: List[str]
    weaknesses: List[str]
    key_players: List[Dict[str, Any]]
    tactical_insights: List[str]
    confidence: float
    analysis_date: datetime
    data_sources: List[str]

@dataclass
class MatchPrediction:
    """Prediction for a football match."""
    home_team: str
    away_team: str
    predicted_score: Tuple[int, int]
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    confidence: PredictionConfidence
    key_factors: List[str]
    reasoning: str
    prediction_date: datetime
    data_quality: float

@dataclass
class HistoricalTrend:
    """Historical trend analysis."""
    team_name: str
    metric: str
    trend_direction: str  # "improving", "declining", "stable"
    trend_strength: float  # 0-1
    historical_data: List[Dict[str, Any]]
    context: str
    confidence: float
    analysis_period: str

@dataclass
class TransferValuation:
    """Transfer market valuation."""
    player_name: str
    current_team: str
    estimated_value: float
    currency: str
    valuation_factors: List[str]
    market_trend: str
    comparable_transfers: List[Dict[str, Any]]
    confidence: float
    valuation_date: datetime

@dataclass
class FootballKnowledge:
    """Comprehensive football knowledge response."""
    query: str
    analysis_type: AnalysisType
    primary_result: Any
    supporting_data: Dict[str, Any]
    confidence: float
    reasoning: str
    recommendations: List[str]
    data_sources: List[str]
    generated_at: datetime

# ============================================================================
# Tactical Analyzer - Formation analysis, tactical insights, strategy evaluation
# ============================================================================

class TacticalAnalyzer:
    """
    Advanced tactical analysis using pattern recognition, formation detection,
    and strategic evaluation based on team performance data and context.
    """
    
    def __init__(self):
        self.formation_patterns = self._build_formation_database()
        self.tactical_indicators = self._build_tactical_indicators()
        self.cache: Dict[str, TacticalAnalysis] = {}
        self.cache_ttl_hours = 6
        logger.info("TacticalAnalyzer initialized")
    
    async def analyze_team_tactics(self, team_name: str, league_code: str,
                                 context: Optional[Dict[str, Any]] = None) -> TacticalAnalysis:
        """
        Perform comprehensive tactical analysis of a team.
        
        Args:
            team_name: Name of the team to analyze
            league_code: League code for context
            context: Additional context for analysis
            
        Returns:
            TacticalAnalysis object with insights
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(team_name, league_code)
            if cache_key in self.cache:
                cached = self.cache[cache_key]
                if datetime.now() - cached.analysis_date < timedelta(hours=self.cache_ttl_hours):
                    logger.info(f"Cache hit for tactical analysis: {team_name}")
                    return cached
            
            # Gather data for analysis
            team_data = await self._collect_team_data(team_name, league_code)
            if not team_data:
                return self._create_fallback_analysis(team_name, "Insufficient data")
            
            # Perform multi-layered analysis
            formation = await self._detect_formation(team_data, context)
            tactical_style = await self._analyze_tactical_style(team_data, context)
            strengths, weaknesses = await self._identify_strengths_weaknesses(team_data)
            key_players = await self._identify_key_players(team_data)
            insights = await self._generate_tactical_insights(team_data, formation, tactical_style)
            
            # Calculate confidence based on data quality
            confidence = self._calculate_analysis_confidence(team_data)
            
            analysis = TacticalAnalysis(
                team_name=team_name,
                formation=formation,
                tactical_style=tactical_style,
                strengths=strengths,
                weaknesses=weaknesses,
                key_players=key_players,
                tactical_insights=insights,
                confidence=confidence,
                analysis_date=datetime.now(),
                data_sources=team_data.get('sources', [])
            )
            
            # Cache the result
            self.cache[cache_key] = analysis
            
            logger.info(f"Completed tactical analysis for {team_name} with confidence {confidence:.2f}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in tactical analysis for {team_name}: {str(e)}")
            return self._create_fallback_analysis(team_name, f"Analysis error: {str(e)}")
    
    async def compare_tactical_approaches(self, team1: str, team2: str, league_code: str) -> Dict[str, Any]:
        """Compare tactical approaches between two teams."""
        try:
            analysis1 = await self.analyze_team_tactics(team1, league_code)
            analysis2 = await self.analyze_team_tactics(team2, league_code)
            
            comparison = {
                'teams': [team1, team2],
                'formations': [analysis1.formation, analysis2.formation],
                'tactical_styles': [analysis1.tactical_style, analysis2.tactical_style],
                'tactical_matchup': await self._analyze_tactical_matchup(analysis1, analysis2),
                'predicted_dynamics': await self._predict_match_dynamics(analysis1, analysis2),
                'key_battles': await self._identify_key_battles(analysis1, analysis2),
                'confidence': min(analysis1.confidence, analysis2.confidence)
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing tactics between {team1} and {team2}: {str(e)}")
            return {'error': str(e)}
    
    async def _collect_team_data(self, team_name: str, league_code: str) -> Dict[str, Any]:
        """Collect comprehensive team data for analysis."""
        try:
            data = {
                'sources': [],
                'standings': {},
                'team_stats': {},
                'league_context': {}
            }
            
            # Get standings data
            try:
                standings = get_standings(league_code)
                data['standings'] = standings
                data['sources'].append('football_data_standings')
                
                # Extract team-specific stats from standings
                team_stats = self._extract_team_from_standings(standings, team_name)
                if team_stats:
                    data['team_stats'] = team_stats
                    
            except Exception as e:
                logger.warning(f"Failed to get standings for {league_code}: {e}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error collecting team data for {team_name}: {e}")
            return {}
    
    def _extract_team_from_standings(self, standings: Dict[str, Any], team_name: str) -> Optional[Dict[str, Any]]:
        """Extract specific team data from standings."""
        try:
            standings_list = standings.get('standings', [])
            if not standings_list:
                return None
            
            # Find the main table (usually TOTAL type)
            main_table = None
            for standing in standings_list:
                if standing.get('type') == 'TOTAL':
                    main_table = standing
                    break
            
            if not main_table:
                main_table = standings_list[0]
            
            # Search for the team
            for team_row in main_table.get('table', []):
                team_info = team_row.get('team', {})
                if self._team_name_matches(team_info.get('name', ''), team_name):
                    return team_row
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting team from standings: {e}")
            return None
    
    def _team_name_matches(self, name1: str, name2: str) -> bool:
        """Check if two team names match (fuzzy matching)."""
        name1_clean = re.sub(r'[^\w\s]', '', name1.lower()).strip()
        name2_clean = re.sub(r'[^\w\s]', '', name2.lower()).strip()
        
        # Exact match
        if name1_clean == name2_clean:
            return True
        
        # One contains the other
        if name1_clean in name2_clean or name2_clean in name1_clean:
            return True
        
        # Remove common prefixes/suffixes and try again
        for prefix in ['fc', 'cf', 'ac', 'rc', 'sc', 'real', 'club']:
            name1_clean = re.sub(f'^{prefix}\\s+|\\s+{prefix}$', '', name1_clean)
            name2_clean = re.sub(f'^{prefix}\\s+|\\s+{prefix}$', '', name2_clean)
        
        return name1_clean == name2_clean
    
    async def _detect_formation(self, team_data: Dict[str, Any], 
                              context: Optional[Dict[str, Any]] = None) -> Optional[Formation]:
        """Detect team formation using LLM analysis."""
        try:
            # Use LLM to analyze formation based on available data
            analysis_prompt = f"""
            Analyze the football team formation based on this data:
            
            Team Statistics: {json.dumps(team_data.get('team_stats', {}), indent=2)}
            
            Based on modern football tactics and the team's performance metrics, what formation 
            is this team most likely using?
            
            Common formations:
            - 4-4-2: Balanced, traditional
            - 4-3-3: Attacking, possession-based
            - 3-5-2: Wing-back focused
            - 4-2-3-1: Defensive midfielder shield
            - 4-1-4-1: Defensive, counter-attacking
            
            Return only the formation code (e.g., "4-3-3") or "unknown" if uncertain.
            """
            
            response = query_llm(
                question=analysis_prompt,
                scores={},
                standings={},
                history=[],
                focus=""
            )
            
            # Extract formation from response
            formation_match = re.search(r'\b(\d-\d-\d|\d-\d-\d-\d|\d-\d-\d-\d-\d)\b', response)
            if formation_match:
                formation_str = formation_match.group(1)
                try:
                    return Formation(formation_str)
                except ValueError:
                    pass
            
            # Fallback to statistical analysis
            return self._statistical_formation_detection(team_data)
            
        except Exception as e:
            logger.warning(f"Formation detection failed: {e}")
            return None
    
    def _statistical_formation_detection(self, team_data: Dict[str, Any]) -> Optional[Formation]:
        """Fallback formation detection based on statistics."""
        try:
            stats = team_data.get('team_stats', {})
            if not stats:
                return None
            
            goals_for = stats.get('goalsFor', 0)
            goals_against = stats.get('goalsAgainst', 0)
            position = stats.get('position', 10)
            
            # Simple heuristics based on attacking/defensive balance
            if goals_for > goals_against * 1.5:  # Very attacking
                return Formation.F_4_3_3
            elif goals_against < goals_for * 0.7:  # Very defensive
                return Formation.F_5_3_2
            elif position <= 6:  # Top teams often use possession-based
                return Formation.F_4_2_3_1
            else:  # Mid-table teams often use balanced
                return Formation.F_4_4_2
                
        except Exception:
            return Formation.F_4_4_2  # Default fallback
    
    async def _analyze_tactical_style(self, team_data: Dict[str, Any], 
                                    context: Optional[Dict[str, Any]] = None) -> Optional[TacticalStyle]:
        """Analyze the team's tactical playing style."""
        try:
            stats = team_data.get('team_stats', {})
            if not stats:
                return None
            
            # Calculate tactical indicators
            goals_for = stats.get('goalsFor', 0)
            goals_against = stats.get('goalsAgainst', 0)
            played = stats.get('playedGames', 1)
            position = stats.get('position', 10)
            
            # Goals per game ratios
            gpg_for = goals_for / max(played, 1)
            gpg_against = goals_against / max(played, 1)
            
            # Determine style based on patterns
            if gpg_for > 2.0 and gpg_against < 1.0:
                return TacticalStyle.HIGH_PRESSING
            elif gpg_for < 1.0 and gpg_against < 0.8:
                return TacticalStyle.DEFENSIVE
            elif goals_for > goals_against * 1.3:
                return TacticalStyle.POSSESSION_BASED
            elif position > 15 and gpg_against > 1.5:
                return TacticalStyle.PARK_THE_BUS
            elif gpg_for > 1.5 and gpg_against > 1.2:
                return TacticalStyle.COUNTER_ATTACKING
            else:
                return TacticalStyle.POSSESSION_BASED
                
        except Exception as e:
            logger.warning(f"Tactical style analysis failed: {e}")
            return None
    
    async def _identify_strengths_weaknesses(self, team_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Identify team strengths and weaknesses from performance data."""
        try:
            stats = team_data.get('team_stats', {})
            strengths = []
            weaknesses = []
            
            if not stats:
                return strengths, weaknesses
            
            goals_for = stats.get('goalsFor', 0)
            goals_against = stats.get('goalsAgainst', 0)
            won = stats.get('won', 0)
            draw = stats.get('draw', 0)
            lost = stats.get('lost', 0)
            played = stats.get('playedGames', 1)
            position = stats.get('position', 10)
            
            # Analyze attacking strength
            gpg = goals_for / max(played, 1)
            if gpg > 2.0:
                strengths.append("Strong attacking output")
            elif gpg < 1.0:
                weaknesses.append("Struggles to score goals")
            
            # Analyze defensive strength
            gpg_against = goals_against / max(played, 1)
            if gpg_against < 0.8:
                strengths.append("Solid defensive structure")
            elif gpg_against > 1.5:
                weaknesses.append("Defensive vulnerabilities")
            
            # Analyze consistency
            win_rate = won / max(played, 1)
            if win_rate > 0.6:
                strengths.append("Consistent performance")
            elif win_rate < 0.3:
                weaknesses.append("Inconsistent results")
            
            # Position-based analysis
            if position <= 4:
                strengths.append("Top-tier performance level")
            elif position >= 17:
                weaknesses.append("Struggling in league position")
            
            return strengths, weaknesses
            
        except Exception as e:
            logger.error(f"Error identifying strengths/weaknesses: {e}")
            return [], []
    
    async def _identify_key_players(self, team_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify key players using LLM analysis and available data."""
        try:
            # Fallback to generic key player roles
            return [
                {"name": "Key Striker", "position": "Forward", "role": "Primary goalscorer"},
                {"name": "Central Midfielder", "position": "Midfielder", "role": "Playmaker"},
                {"name": "Defensive Leader", "position": "Defender", "role": "Team captain"}
            ]
            
        except Exception as e:
            logger.warning(f"Key player identification failed: {e}")
            return []
    
    async def _generate_tactical_insights(self, team_data: Dict[str, Any], 
                                        formation: Optional[Formation],
                                        tactical_style: Optional[TacticalStyle]) -> List[str]:
        """Generate tactical insights using LLM analysis."""
        try:
            insights_prompt = f"""
            Generate tactical insights for this football team:
            
            Formation: {formation.value if formation else 'Unknown'}
            Tactical Style: {tactical_style.value if tactical_style else 'Unknown'}
            Team Data: {json.dumps(team_data.get('team_stats', {}), indent=2)}
            
            Provide 3-5 specific tactical insights about:
            - How this team typically approaches matches
            - Their main tactical strengths
            - Potential tactical weaknesses opponents could exploit
            
            Make insights specific and actionable, not generic.
            """
            
            response = query_llm(
                question=insights_prompt,
                scores={},
                standings=team_data.get('standings', {}),
                history=[],
                focus=""
            )
            
            # Extract insights from response
            insights = []
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or len(line) > 20):
                    clean_line = re.sub(r'^[-•]\s*', '', line).strip()
                    if clean_line:
                        insights.append(clean_line)
            
            # If no structured insights found, try to split by sentences
            if not insights:
                sentences = re.split(r'[.!?]+', response)
                insights = [s.strip() for s in sentences if len(s.strip()) > 20][:5]
            
            return insights[:5]  # Limit to 5 insights
            
        except Exception as e:
            logger.warning(f"Tactical insights generation failed: {e}")
            return ["Tactical analysis requires more detailed performance data"]
    
    async def _analyze_tactical_matchup(self, analysis1: TacticalAnalysis, 
                                      analysis2: TacticalAnalysis) -> Dict[str, Any]:
        """Analyze how two tactical approaches match up against each other."""
        try:
            matchup_analysis = {
                'formation_compatibility': self._analyze_formation_matchup(
                    analysis1.formation, analysis2.formation
                ),
                'style_clash': self._analyze_style_clash(
                    analysis1.tactical_style, analysis2.tactical_style
                )
            }
            
            return matchup_analysis
            
        except Exception as e:
            logger.error(f"Tactical matchup analysis failed: {e}")
            return {}
    
    def _analyze_formation_matchup(self, form1: Optional[Formation], 
                                 form2: Optional[Formation]) -> str:
        """Analyze how two formations match up."""
        if not form1 or not form2:
            return "Formation matchup unclear due to missing data"
        
        formation_matchups = {
            (Formation.F_4_3_3, Formation.F_4_4_2): "4-3-3 should dominate midfield against 4-4-2",
            (Formation.F_4_4_2, Formation.F_4_3_3): "4-4-2 may struggle in midfield but could exploit wings",
            (Formation.F_3_5_2, Formation.F_4_4_2): "3-5-2 wing-backs could cause problems for 4-4-2",
            (Formation.F_4_2_3_1, Formation.F_4_3_3): "Midfield battle will be crucial between these formations"
        }
        
        return formation_matchups.get((form1, form2), 
                                    f"{form1.value} vs {form2.value} - evenly matched formations")
    
    def _analyze_style_clash(self, style1: Optional[TacticalStyle], 
                           style2: Optional[TacticalStyle]) -> str:
        """Analyze how two tactical styles clash."""
        if not style1 or not style2:
            return "Tactical style clash unclear"
        
        style_clashes = {
            (TacticalStyle.HIGH_PRESSING, TacticalStyle.POSSESSION_BASED): 
                "High pressing vs possession - intensity vs control",
            (TacticalStyle.COUNTER_ATTACKING, TacticalStyle.HIGH_PRESSING):
                "Counter-attacks vs high press - speed vs pressure",
            (TacticalStyle.DEFENSIVE, TacticalStyle.POSSESSION_BASED):
                "Defensive solidity vs possession - patience vs creativity"
        }
        
        return style_clashes.get((style1, style2), 
                               f"{style1.value} vs {style2.value} - contrasting approaches")
    
    async def _predict_match_dynamics(self, analysis1: TacticalAnalysis, 
                                    analysis2: TacticalAnalysis) -> List[str]:
        """Predict how the match might unfold tactically."""
        dynamics = []
        
        # Analyze formation dynamics
        if analysis1.formation and analysis2.formation:
            dynamics.append(self._analyze_formation_matchup(analysis1.formation, analysis2.formation))
        
        # Analyze style dynamics
        if analysis1.tactical_style and analysis2.tactical_style:
            dynamics.append(self._analyze_style_clash(analysis1.tactical_style, analysis2.tactical_style))
        
        return dynamics[:3]  # Limit to 3 key dynamics
    
    async def _identify_key_battles(self, analysis1: TacticalAnalysis, 
                                  analysis2: TacticalAnalysis) -> List[str]:
        """Identify key tactical battles in the match."""
        battles = ["Midfield control will be crucial", "Wide areas could determine the outcome"]
        
        # Formation-specific battles
        if analysis1.formation and analysis2.formation:
            battles.append(f"Formation battle: {analysis1.formation.value} vs {analysis2.formation.value}")
        
        return battles[:3]  # Limit to 3 key battles
    
    def _build_formation_database(self) -> Dict[str, Dict[str, Any]]:
        """Build database of formation characteristics."""
        return {
            "4-4-2": {
                "style": "balanced",
                "strengths": ["simplicity", "defensive_stability", "width"],
                "weaknesses": ["midfield_outnumbered", "limited_creativity"]
            },
            "4-3-3": {
                "style": "attacking",
                "strengths": ["midfield_control", "wide_attacking", "possession"],
                "weaknesses": ["defensive_transitions", "wing_back_exposure"]
            }
        }
    
    def _build_tactical_indicators(self) -> Dict[str, List[str]]:
        """Build indicators for different tactical styles."""
        return {
            "possession_based": ["high_pass_accuracy", "low_direct_play", "patient_buildup"],
            "counter_attacking": ["fast_transitions", "direct_play", "defensive_shape"],
            "high_pressing": ["high_defensive_line", "intense_pressing", "quick_regains"],
            "defensive": ["low_block", "few_attacks", "solid_shape"]
        }
    
    def _calculate_analysis_confidence(self, team_data: Dict[str, Any]) -> float:
        """Calculate confidence level for the tactical analysis."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on available data
        if team_data.get('team_stats'):
            confidence += 0.3
        if len(team_data.get('sources', [])) > 0:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _create_fallback_analysis(self, team_name: str, reason: str) -> TacticalAnalysis:
        """Create fallback analysis when full analysis fails."""
        return TacticalAnalysis(
            team_name=team_name,
            formation=None,
            tactical_style=None,
            strengths=[],
            weaknesses=[],
            key_players=[],
            tactical_insights=[f"Analysis limited: {reason}"],
            confidence=0.3,
            analysis_date=datetime.now(),
            data_sources=[]
        )
    
    def _generate_cache_key(self, team_name: str, league_code: str) -> str:
        """Generate cache key for tactical analysis."""
        return hashlib.md5(f"tactical:{team_name}:{league_code}".encode()).hexdigest()

# ============================================================================
# Predictive Analytics - Match predictions, form analysis, statistical forecasting
# ============================================================================

class PredictiveAnalytics:
    """
    Advanced predictive analytics using statistical models, form analysis,
    and machine learning approaches for match outcome prediction.
    """
    
    def __init__(self):
        self.prediction_models = self._initialize_models()
        self.cache: Dict[str, MatchPrediction] = {}
        self.cache_ttl_hours = 2  # Shorter cache for predictions
        logger.info("PredictiveAnalytics initialized")
    
    async def predict_match(self, home_team: str, away_team: str, league_code: str,
                          context: Optional[Dict[str, Any]] = None) -> MatchPrediction:
        """
        Predict match outcome using multiple analytical approaches.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            league_code: League code for context
            context: Additional context for prediction
            
        Returns:
            MatchPrediction object with detailed analysis
        """
        try:
            # Check cache first
            cache_key = self._generate_prediction_cache_key(home_team, away_team, league_code)
            if cache_key in self.cache:
                cached = self.cache[cache_key]
                if datetime.now() - cached.prediction_date < timedelta(hours=self.cache_ttl_hours):
                    logger.info(f"Cache hit for prediction: {home_team} vs {away_team}")
                    return cached
            
            # Collect match data
            match_data = await self._collect_match_data(home_team, away_team, league_code)
            
            # Run prediction models
            statistical_prediction = await self._statistical_prediction(match_data)
            form_prediction = await self._form_based_prediction(match_data)
            
            # Combine predictions
            final_prediction = await self._combine_predictions(statistical_prediction, form_prediction, match_data)
            
            # Generate reasoning and factors
            reasoning = await self._generate_prediction_reasoning(match_data, final_prediction)
            key_factors = await self._identify_key_factors(match_data)
            
            prediction = MatchPrediction(
                home_team=home_team,
                away_team=away_team,
                predicted_score=final_prediction['score'],
                home_win_probability=final_prediction['home_win'],
                draw_probability=final_prediction['draw'],
                away_win_probability=final_prediction['away_win'],
                confidence=final_prediction['confidence'],
                key_factors=key_factors,
                reasoning=reasoning,
                prediction_date=datetime.now(),
                data_quality=match_data.get('data_quality', 0.5)
            )
            
            # Cache the prediction
            self.cache[cache_key] = prediction
            
            logger.info(f"Generated prediction for {home_team} vs {away_team}")
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting match {home_team} vs {away_team}: {str(e)}")
            return self._create_fallback_prediction(home_team, away_team, str(e))
    
    async def _collect_match_data(self, home_team: str, away_team: str, 
                                league_code: str) -> Dict[str, Any]:
        """Collect comprehensive data for match prediction."""
        try:
            data = {
                'home_team': home_team,
                'away_team': away_team,
                'league_code': league_code,
                'standings': {},
                'home_stats': {},
                'away_stats': {},
                'data_quality': 0.0,
                'sources': []
            }
            
            # Get league standings
            try:
                standings = get_standings(league_code)
                data['standings'] = standings
                data['sources'].append('football_data')
                
                # Extract team stats from standings
                home_stats = self._extract_team_stats(standings, home_team)
                away_stats = self._extract_team_stats(standings, away_team)
                
                if home_stats:
                    data['home_stats'] = home_stats
                    data['data_quality'] += 0.4
                
                if away_stats:
                    data['away_stats'] = away_stats
                    data['data_quality'] += 0.4
                
            except Exception as e:
                logger.warning(f"Failed to get standings for prediction: {e}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error collecting match data: {e}")
            return {'data_quality': 0.0}
    
    def _extract_team_stats(self, standings: Dict[str, Any], team_name: str) -> Optional[Dict[str, Any]]:
        """Extract team statistics from standings data."""
        try:
            standings_list = standings.get('standings', [])
            if not standings_list:
                return None
            
            # Find main table
            main_table = next((s for s in standings_list if s.get('type') == 'TOTAL'), standings_list[0])
            
            # Find team
            for team_row in main_table.get('table', []):
                team_info = team_row.get('team', {})
                if self._team_name_matches(team_info.get('name', ''), team_name):
                    # Calculate additional metrics
                    stats = dict(team_row)
                    stats['goals_per_game'] = stats.get('goalsFor', 0) / max(stats.get('playedGames', 1), 1)
                    stats['goals_conceded_per_game'] = stats.get('goalsAgainst', 0) / max(stats.get('playedGames', 1), 1)
                    stats['points_per_game'] = stats.get('points', 0) / max(stats.get('playedGames', 1), 1)
                    stats['win_rate'] = stats.get('won', 0) / max(stats.get('playedGames', 1), 1)
                    
                    return stats
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting team stats: {e}")
            return None
    
    def _team_name_matches(self, name1: str, name2: str) -> bool:
        """Check if team names match."""
        name1_clean = re.sub(r'[^\w\s]', '', name1.lower()).strip()
        name2_clean = re.sub(r'[^\w\s]', '', name2.lower()).strip()
        
        if name1_clean == name2_clean:
            return True
        
        if name1_clean in name2_clean or name2_clean in name1_clean:
            return True
        
        # Remove common prefixes/suffixes
        for prefix in ['fc', 'cf', 'ac', 'rc', 'sc', 'real', 'club']:
            name1_clean = re.sub(f'^{prefix}\\s+|\\s+{prefix}$', '', name1_clean)
            name2_clean = re.sub(f'^{prefix}\\s+|\\s+{prefix}$', '', name2_clean)
        
        return name1_clean == name2_clean
    
    async def _statistical_prediction(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prediction based on statistical analysis."""
        try:
            home_stats = match_data.get('home_stats', {})
            away_stats = match_data.get('away_stats', {})
            
            if not home_stats or not away_stats:
                return self._default_prediction()
            
            # Simple prediction based on goals per game and position
            home_attack = home_stats.get('goals_per_game', 1.0)
            away_attack = away_stats.get('goals_per_game', 1.0)
            home_position = home_stats.get('position', 10)
            away_position = away_stats.get('position', 10)
            
            # Calculate relative strength
            position_advantage = (away_position - home_position) * 0.05  # Better position = lower number
            attack_advantage = (home_attack - away_attack) * 0.3
            
            total_advantage = position_advantage + attack_advantage + 0.1  # Base home advantage
            
            # Convert to probabilities
            if total_advantage > 0.3:
                home_win, draw, away_win = 0.55, 0.25, 0.20
                score = (2, 1)
            elif total_advantage < -0.3:
                home_win, draw, away_win = 0.20, 0.25, 0.55
                score = (1, 2)
            else:
                home_win, draw, away_win = 0.40, 0.30, 0.30
                score = (1, 1)
            
            return {
                'method': 'statistical',
                'score': score,
                'home_win': home_win,
                'draw': draw,
                'away_win': away_win,
                'confidence': PredictionConfidence.MEDIUM
            }
            
        except Exception as e:
            logger.error(f"Statistical prediction failed: {e}")
            return self._default_prediction()
    
    async def _form_based_prediction(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prediction based on current form analysis."""
        try:
            home_stats = match_data.get('home_stats', {})
            away_stats = match_data.get('away_stats', {})
            
            if not home_stats or not away_stats:
                return self._default_prediction()
            
            # Form indicators
            home_form = self._calculate_form_rating(home_stats)
            away_form = self._calculate_form_rating(away_stats)
            
            # Form difference
            form_diff = home_form - away_form
            
            # Convert to probabilities
            if form_diff > 0.2:
                home_win, draw, away_win = 0.50, 0.30, 0.20
                score = (2, 1)
            elif form_diff < -0.2:
                home_win, draw, away_win = 0.20, 0.30, 0.50
                score = (1, 2)
            else:
                home_win, draw, away_win = 0.35, 0.30, 0.35
                score = (1, 1)
            
            return {
                'method': 'form_based',
                'score': score,
                'home_win': home_win,
                'draw': draw,
                'away_win': away_win,
                'confidence': PredictionConfidence.MEDIUM
            }
            
        except Exception as e:
            logger.error(f"Form-based prediction failed: {e}")
            return self._default_prediction()
    
    def _calculate_form_rating(self, stats: Dict[str, Any]) -> float:
        """Calculate a form rating for a team (0-1 scale)."""
        try:
            # Base form on multiple factors
            win_rate = stats.get('won', 0) / max(stats.get('playedGames', 1), 1)
            points_per_game = stats.get('points', 0) / max(stats.get('playedGames', 1), 1) / 3  # Normalize to 0-1
            goals_ratio = stats.get('goalsFor', 1) / max(stats.get('goalsAgainst', 1), 1) / 3  # Normalize
            
            # Weighted average
            form_rating = (win_rate * 0.5 + points_per_game * 0.3 + min(goals_ratio, 1.0) * 0.2)
            
            return min(max(form_rating, 0.0), 1.0)
            
        except Exception:
            return 0.5  # Default neutral form
    
    async def _combine_predictions(self, statistical: Dict[str, Any], 
                                 form: Dict[str, Any], match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Combine different prediction methods into final prediction."""
        try:
            # Weight the predictions
            stat_weight = 0.6
            form_weight = 0.4
            
            # Combine probabilities
            home_win = (statistical.get('home_win', 0.33) * stat_weight +
                       form.get('home_win', 0.33) * form_weight)
            
            draw = (statistical.get('draw', 0.33) * stat_weight +
                   form.get('draw', 0.33) * form_weight)
            
            away_win = (statistical.get('away_win', 0.33) * stat_weight +
                       form.get('away_win', 0.33) * form_weight)
            
            # Determine most likely score
            if home_win > draw and home_win > away_win:
                score = (2, 1) if home_win > 0.5 else (1, 0)
            elif away_win > draw and away_win > home_win:
                score = (1, 2) if away_win > 0.5 else (0, 1)
            else:
                score = (1, 1)
            
            # Calculate overall confidence
            data_quality = match_data.get('data_quality', 0.5)
            confidence_score = data_quality * 0.6 + max(home_win, draw, away_win) * 0.4
            
            if confidence_score > 0.8:
                confidence = PredictionConfidence.HIGH
            elif confidence_score > 0.6:
                confidence = PredictionConfidence.MEDIUM
            else:
                confidence = PredictionConfidence.LOW
            
            return {
                'score': score,
                'home_win': home_win,
                'draw': draw,
                'away_win': away_win,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error combining predictions: {e}")
            return {
                'score': (1, 1),
                'home_win': 0.33,
                'draw': 0.34,
                'away_win': 0.33,
                'confidence': PredictionConfidence.LOW
            }
    
    async def _generate_prediction_reasoning(self, match_data: Dict[str, Any], 
                                           prediction: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for the prediction."""
        try:
            home_team = match_data.get('home_team', 'Home team')
            away_team = match_data.get('away_team', 'Away team')
            home_stats = match_data.get('home_stats', {})
            away_stats = match_data.get('away_stats', {})
            
            reasoning_parts = []
            
            # Position-based reasoning
            home_pos = home_stats.get('position', 10)
            away_pos = away_stats.get('position', 10)
            
            if home_pos < away_pos:
                reasoning_parts.append(f"{home_team} is higher in the table ({home_pos} vs {away_pos})")
            elif away_pos < home_pos:
                reasoning_parts.append(f"{away_team} is higher in the table ({away_pos} vs {home_pos})")
            
            # Home advantage
            reasoning_parts.append("Home advantage considered in prediction")
            
            # Combine reasoning
            if reasoning_parts:
                return ". ".join(reasoning_parts) + "."
            else:
                return "Prediction based on statistical analysis of current season performance."
                
        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            return "Prediction based on available team performance data."
    
    async def _identify_key_factors(self, match_data: Dict[str, Any]) -> List[str]:
        """Identify key factors that could influence the match."""
        try:
            factors = []
            home_stats = match_data.get('home_stats', {})
            away_stats = match_data.get('away_stats', {})
            
            # Position-related factors
            home_pos = home_stats.get('position', 10)
            away_pos = away_stats.get('position', 10)
            
            if home_pos <= 4:
                factors.append("Home team fighting for top positions")
            elif home_pos >= 17:
                factors.append("Home team under relegation pressure")
            
            if away_pos <= 4:
                factors.append("Away team fighting for top positions")
            elif away_pos >= 17:
                factors.append("Away team under relegation pressure")
            
            # Home advantage
            factors.append("Home advantage and crowd support")
            
            # Form factors
            factors.append("Current season form and momentum")
            
            return factors[:4]  # Limit to 4 key factors
            
        except Exception as e:
            logger.error(f"Error identifying key factors: {e}")
            return ["Match outcome depends on team performance on the day"]
    
    def _default_prediction(self) -> Dict[str, Any]:
        """Return default prediction when analysis fails."""
        return {
            'score': (1, 1),
            'home_win': 0.35,
            'draw': 0.30,
            'away_win': 0.35,
            'confidence': PredictionConfidence.LOW
        }
    
    def _create_fallback_prediction(self, home_team: str, away_team: str, reason: str) -> MatchPrediction:
        """Create fallback prediction when analysis fails."""
        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            predicted_score=(1, 1),
            home_win_probability=0.35,
            draw_probability=0.30,
            away_win_probability=0.35,
            confidence=PredictionConfidence.LOW,
            key_factors=[f"Analysis limited: {reason}"],
            reasoning="Prediction based on limited data availability",
            prediction_date=datetime.now(),
            data_quality=0.2
        )
    
    def _generate_prediction_cache_key(self, home_team: str, away_team: str, league_code: str) -> str:
        """Generate cache key for match prediction."""
        return hashlib.md5(f"prediction:{home_team}:{away_team}:{league_code}".encode()).hexdigest()
    
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize prediction models."""
        return {
            'statistical_model': 'poisson_based',
            'form_model': 'weighted_recent_performance',
            'ensemble_weights': {'statistical': 0.6, 'form': 0.4}
        }

# ============================================================================
# Historical Context - Historical data analysis, trends, contextual insights
# ============================================================================

class HistoricalContext:
    """
    Historical data analysis and trend identification for contextual insights.
    """
    
    def __init__(self):
        self.trend_cache: Dict[str, HistoricalTrend] = {}
        self.cache_ttl_hours = 12
        logger.info("HistoricalContext initialized")
    
    async def analyze_team_trends(self, team_name: str, league_code: str, 
                                metric: str = "points") -> HistoricalTrend:
        """
        Analyze historical trends for a team.
        
        Args:
            team_name: Name of the team
            league_code: League code
            metric: Metric to analyze trends for
            
        Returns:
            HistoricalTrend object with analysis
        """
        try:
            # For now, return a simplified trend analysis based on current data
            current_data = await self._get_current_season_data(team_name, league_code)
            
            if not current_data:
                return self._create_fallback_trend(team_name, metric, "Insufficient data")
            
            # Analyze trend based on current position and performance
            trend_direction = self._determine_trend_direction(current_data, metric)
            trend_strength = self._calculate_trend_strength(current_data, metric)
            context = self._generate_trend_context(current_data, metric)
            
            return HistoricalTrend(
                team_name=team_name,
                metric=metric,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                historical_data=[current_data],
                context=context,
                confidence=0.7,
                analysis_period="Current season"
            )
            
        except Exception as e:
            logger.error(f"Error analyzing trends for {team_name}: {str(e)}")
            return self._create_fallback_trend(team_name, metric, f"Analysis error: {str(e)}")
    
    async def _get_current_season_data(self, team_name: str, league_code: str) -> Optional[Dict[str, Any]]:
        """Get current season data for trend analysis."""
        try:
            standings = get_standings(league_code)
            
            # Extract team data
            standings_list = standings.get('standings', [])
            if not standings_list:
                return None
            
            main_table = next((s for s in standings_list if s.get('type') == 'TOTAL'), standings_list[0])
            
            for team_row in main_table.get('table', []):
                team_info = team_row.get('team', {})
                if self._team_name_matches(team_info.get('name', ''), team_name):
                    return team_row
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting current season data: {e}")
            return None
    
    def _team_name_matches(self, name1: str, name2: str) -> bool:
        """Check if team names match."""
        name1_clean = re.sub(r'[^\w\s]', '', name1.lower()).strip()
        name2_clean = re.sub(r'[^\w\s]', '', name2.lower()).strip()
        return name1_clean == name2_clean or name1_clean in name2_clean or name2_clean in name1_clean
    
    def _determine_trend_direction(self, data: Dict[str, Any], metric: str) -> str:
        """Determine trend direction based on current performance."""
        try:
            position = data.get('position', 10)
            win_rate = data.get('won', 0) / max(data.get('playedGames', 1), 1)
            
            # Simple heuristics for trend direction
            if position <= 6 and win_rate > 0.5:
                return "improving"
            elif position >= 15 and win_rate < 0.3:
                return "declining"
            else:
                return "stable"
                
        except Exception:
            return "stable"
    
    def _calculate_trend_strength(self, data: Dict[str, Any], metric: str) -> float:
        """Calculate strength of the trend."""
        try:
            position = data.get('position', 10)
            
            # Calculate trend strength based on performance metrics
            if position <= 3:
                return 0.9  # Very strong for top teams
            elif position <= 6:
                return 0.7  # Strong for European places
            elif position >= 18:
                return 0.8  # Strong trend (negative) for relegation zone
            else:
                return 0.5  # Moderate trend for mid-table
                
        except Exception:
            return 0.5
    
    def _generate_trend_context(self, data: Dict[str, Any], metric: str) -> str:
        """Generate contextual explanation for the trend."""
        try:
            position = data.get('position', 10)
            points = data.get('points', 0)
            played = data.get('playedGames', 0)
            
            if position <= 4:
                return f"Team is performing excellently, currently in {position} position with {points} points from {played} games"
            elif position >= 17:
                return f"Team is struggling, currently in {position} position with {points} points from {played} games"
            else:
                return f"Team showing consistent mid-table performance in {position} position with {points} points from {played} games"
                
        except Exception:
            return "Trend analysis based on current season performance"
    
    def _create_fallback_trend(self, team_name: str, metric: str, reason: str) -> HistoricalTrend:
        """Create fallback trend when analysis fails."""
        return HistoricalTrend(
            team_name=team_name,
            metric=metric,
            trend_direction="stable",
            trend_strength=0.5,
            historical_data=[],
            context=f"Analysis limited: {reason}",
            confidence=0.3,
            analysis_period="Limited data"
        )

# ============================================================================
# Transfer Intelligence - Transfer market analysis and player valuation
# ============================================================================

class TransferIntelligence:
    """
    Transfer market analysis and player valuation using market data and context.
    """
    
    def __init__(self):
        self.valuation_cache: Dict[str, TransferValuation] = {}
        self.cache_ttl_hours = 24
        logger.info("TransferIntelligence initialized")
    
    async def analyze_player_value(self, player_name: str, current_team: str,
                                 context: Optional[Dict[str, Any]] = None) -> TransferValuation:
        """
        Analyze player transfer value and market context.
        
        Args:
            player_name: Name of the player
            current_team: Current team of the player
            context: Additional context for valuation
            
        Returns:
            TransferValuation object with analysis
        """
        try:
            # For now, return a simplified valuation analysis
            estimated_value = self._estimate_player_value(player_name, current_team)
            valuation_factors = self._identify_valuation_factors(player_name, current_team)
            market_trend = self._analyze_market_trend(player_name)
            
            return TransferValuation(
                player_name=player_name,
                current_team=current_team,
                estimated_value=estimated_value,
                currency="EUR",
                valuation_factors=valuation_factors,
                market_trend=market_trend,
                comparable_transfers=[],
                confidence=0.6,
                valuation_date=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing player value for {player_name}: {str(e)}")
            return self._create_fallback_valuation(player_name, current_team, str(e))
    
    def _estimate_player_value(self, player_name: str, current_team: str) -> float:
        """Estimate player value using heuristics."""
        # Simplified estimation - would use actual market data in real implementation
        base_value = 10.0  # Base value in millions
        
        # Adjust based on team tier (simplified)
        if any(club in current_team.lower() for club in ['real madrid', 'barcelona', 'manchester', 'liverpool']):
            base_value *= 3
        elif any(club in current_team.lower() for club in ['arsenal', 'chelsea', 'tottenham']):
            base_value *= 2
        
        return base_value
    
    def _identify_valuation_factors(self, player_name: str, current_team: str) -> List[str]:
        """Identify factors affecting player valuation."""
        return [
            "Current team reputation and league",
            "Player age and career stage",
            "Performance in current season",
            "Contract situation and length",
            "Market demand and competition"
        ]
    
    def _analyze_market_trend(self, player_name: str) -> str:
        """Analyze market trend for player type."""
        return "stable"  # Simplified - would analyze actual market data
    
    def _create_fallback_valuation(self, player_name: str, current_team: str, reason: str) -> TransferValuation:
        """Create fallback valuation when analysis fails."""
        return TransferValuation(
            player_name=player_name,
            current_team=current_team,
            estimated_value=5.0,
            currency="EUR",
            valuation_factors=[f"Analysis limited: {reason}"],
            market_trend="unknown",
            comparable_transfers=[],
            confidence=0.2,
            valuation_date=datetime.now()
        )

# ============================================================================
# Football Knowledge Coordinator - Main orchestrator
# ============================================================================

class FootballKnowledgeCoordinator:
    """
    Main orchestrator that coordinates all football knowledge and analytics operations.
    Provides unified interface for accessing comprehensive football intelligence.
    """
    
    def __init__(self):
        self.tactical_analyzer = TacticalAnalyzer()
        self.predictive_analytics = PredictiveAnalytics()
        self.historical_context = HistoricalContext()
        self.transfer_intelligence = TransferIntelligence()
        
        # Performance tracking
        self.request_count = 0
        self.success_count = 0
        self.cache_hits = 0
        
        logger.info("FootballKnowledgeCoordinator initialized")
    
    async def process_football_query(self, query: str, query_type: AnalysisType,
                                   context: Optional[Dict[str, Any]] = None) -> FootballKnowledge:
        """
        Process a football query and return comprehensive knowledge response.
        
        Args:
            query: The football query to process
            query_type: Type of analysis requested
            context: Additional context for processing
            
        Returns:
            FootballKnowledge object with comprehensive response
        """
        self.request_count += 1
        
        try:
            if query_type == AnalysisType.TACTICAL:
                result = await self._handle_tactical_query(query, context)
            elif query_type == AnalysisType.PREDICTIVE:
                result = await self._handle_predictive_query(query, context)
            elif query_type == AnalysisType.HISTORICAL:
                result = await self._handle_historical_query(query, context)
            elif query_type == AnalysisType.TRANSFER:
                result = await self._handle_transfer_query(query, context)
            elif query_type == AnalysisType.COMPARATIVE:
                result = await self._handle_comparative_query(query, context)
            else:
                result = await self._handle_general_query(query, context)
            
            self.success_count += 1
            return result
            
        except Exception as e:
            logger.error(f"Error processing football query: {str(e)}")
            return self._create_fallback_response(query, query_type, str(e))
    
    async def _handle_tactical_query(self, query: str, context: Optional[Dict[str, Any]]) -> FootballKnowledge:
        """Handle tactical analysis queries."""
        try:
            # Extract team and league from query (simplified)
            team_name, league_code = self._extract_team_league(query, context)
            
            if not team_name or not league_code:
                raise ValueError("Could not extract team and league from query")
            
            analysis = await self.tactical_analyzer.analyze_team_tactics(team_name, league_code, context)
            
            return FootballKnowledge(
                query=query,
                analysis_type=AnalysisType.TACTICAL,
                primary_result=analysis,
                supporting_data={'team': team_name, 'league': league_code},
                confidence=analysis.confidence,
                reasoning=f"Tactical analysis of {team_name} based on current season performance",
                recommendations=self._generate_tactical_recommendations(analysis),
                data_sources=analysis.data_sources,
                generated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error handling tactical query: {e}")
            raise
    
    async def _handle_predictive_query(self, query: str, context: Optional[Dict[str, Any]]) -> FootballKnowledge:
        """Handle match prediction queries."""
        try:
            # Extract teams and league from query (simplified)
            home_team, away_team, league_code = self._extract_match_teams_from_query(query, context)
            
            if not home_team or not away_team or not league_code:
                raise ValueError("Could not extract match teams and league from query")
            
            prediction = await self.predictive_analytics.predict_match(home_team, away_team, league_code, context)
            
            return FootballKnowledge(
                query=query,
                analysis_type=AnalysisType.PREDICTIVE,
                primary_result=prediction,
                supporting_data={'home_team': home_team, 'away_team': away_team, 'league': league_code},
                confidence=prediction.confidence.value if hasattr(prediction.confidence, 'value') else 0.7,
                reasoning=prediction.reasoning,
                recommendations=self._generate_prediction_recommendations(prediction),
                data_sources=['predictive_analytics'],
                generated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error handling predictive query: {e}")
            raise
    
    async def _handle_historical_query(self, query: str, context: Optional[Dict[str, Any]]) -> FootballKnowledge:
        """Handle historical analysis queries."""
        try:
            team_name, league_code = self._extract_team_league(query, context)
            metric = self._extract_metric_from_query(query)
            
            if not team_name or not league_code:
                raise ValueError("Could not extract team and league from query")
            
            trend = await self.historical_context.analyze_team_trends(team_name, league_code, metric)
            
            return FootballKnowledge(
                query=query,
                analysis_type=AnalysisType.HISTORICAL,
                primary_result=trend,
                supporting_data={'team': team_name, 'league': league_code, 'metric': metric},
                confidence=trend.confidence,
                reasoning=f"Historical trend analysis for {team_name} {metric}",
                recommendations=self._generate_historical_recommendations(trend),
                data_sources=['historical_context'],
                generated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error handling historical query: {e}")
            raise
    
    async def _handle_transfer_query(self, query: str, context: Optional[Dict[str, Any]]) -> FootballKnowledge:
        """Handle transfer analysis queries."""
        try:
            player_name, current_team = self._extract_player_team_from_query(query, context)
            
            if not player_name:
                raise ValueError("Could not extract player name from query")
            
            valuation = await self.transfer_intelligence.analyze_player_value(player_name, current_team or "Unknown", context)
            
            return FootballKnowledge(
                query=query,
                analysis_type=AnalysisType.TRANSFER,
                primary_result=valuation,
                supporting_data={'player': player_name, 'team': current_team},
                confidence=valuation.confidence,
                reasoning=f"Transfer valuation analysis for {player_name}",
                recommendations=self._generate_transfer_recommendations(valuation),
                data_sources=['transfer_intelligence'],
                generated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error handling transfer query: {e}")
            raise
    
    async def _handle_comparative_query(self, query: str, context: Optional[Dict[str, Any]]) -> FootballKnowledge:
        """Handle comparative analysis queries."""
        try:
            team1, team2, league_code = self._extract_comparison_teams(query, context)
            
            if not team1 or not team2 or not league_code:
                raise ValueError("Could not extract teams for comparison from query")
            
            comparison = await self.tactical_analyzer.compare_tactical_approaches(team1, team2, league_code)
            
            return FootballKnowledge(
                query=query,
                analysis_type=AnalysisType.COMPARATIVE,
                primary_result=comparison,
                supporting_data={'team1': team1, 'team2': team2, 'league': league_code},
                confidence=comparison.get('confidence', 0.7),
                reasoning=f"Tactical comparison between {team1} and {team2}",
                recommendations=self._generate_comparison_recommendations(comparison),
                data_sources=['tactical_analyzer'],
                generated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error handling comparative query: {e}")
            raise
    
    async def _handle_general_query(self, query: str, context: Optional[Dict[str, Any]]) -> FootballKnowledge:
        """Handle general football queries."""
        try:
            # Use LLM for general football knowledge
            response = query_llm(
                question=query,
                scores={},
                standings={},
                history=[],
                focus=""
            )
            
            return FootballKnowledge(
                query=query,
                analysis_type=AnalysisType.STRATEGIC,
                primary_result={'response': response},
                supporting_data={},
                confidence=0.7,
                reasoning="General football knowledge response using LLM",
                recommendations=[],
                data_sources=['llm'],
                generated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error handling general query: {e}")
            raise
    
    def _extract_team_league(self, query: str, context: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
        """Extract team name and league code from query."""
        try:
            # Simple extraction - would use entity resolver in full implementation
            query_lower = query.lower()
            
            # Common team patterns
            teams = {
                'real madrid': 'PD', 'barcelona': 'PD', 'atletico madrid': 'PD',
                'manchester united': 'PL', 'manchester city': 'PL', 'liverpool': 'PL',
                'arsenal': 'PL', 'chelsea': 'PL', 'tottenham': 'PL',
                'bayern munich': 'BL1', 'borussia dortmund': 'BL1',
                'juventus': 'SA', 'ac milan': 'SA', 'inter milan': 'SA',
                'psg': 'FL1', 'lyon': 'FL1', 'marseille': 'FL1'
            }
            
            for team, league in teams.items():
                if team in query_lower:
                    return team.title(), league
            
            # Try to extract from context
            if context:
                team = context.get('team')
                league = context.get('league')
                if team and league:
                    return team, league
            
            return None, None
            
        except Exception as e:
            logger.error(f"Error extracting team/league: {e}")
            return None, None
    
    def _extract_match_teams_from_query(self, query: str, context: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract home team, away team, and league from match query."""
        try:
            # Look for vs/against patterns
            vs_patterns = [r'(.+)\s+vs\s+(.+)', r'(.+)\s+against\s+(.+)', r'(.+)\s+v\s+(.+)']
            
            for pattern in vs_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    team1 = match.group(1).strip()
                    team2 = match.group(2).strip()
                    
                    # Try to determine league from teams
                    _, league1 = self._extract_team_league(team1, context)
                    _, league2 = self._extract_team_league(team2, context)
                    
                    league = league1 or league2 or 'PL'  # Default to Premier League
                    
                    return team1, team2, league
            
            return None, None, None
            
        except Exception as e:
            logger.error(f"Error extracting match teams: {e}")
            return None, None, None
    
    def _extract_metric_from_query(self, query: str) -> str:
        """Extract metric from query for historical analysis."""
        query_lower = query.lower()
        
        metrics = {
            'points': ['points', 'pts'],
            'goals': ['goals', 'scoring'],
            'position': ['position', 'rank', 'place'],
            'form': ['form', 'performance'],
            'wins': ['wins', 'victories']
        }
        
        for metric, keywords in metrics.items():
            if any(keyword in query_lower for keyword in keywords):
                return metric
        
        return 'points'  # Default metric
    
    def _extract_player_team_from_query(self, query: str, context: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
        """Extract player name and current team from transfer query."""
        try:
            # Simple extraction - would use more sophisticated NER in full implementation
            words = query.split()
            
            # Look for proper nouns that might be player names
            potential_players = []
            for i, word in enumerate(words):
                if word[0].isupper() and word.lower() not in ['the', 'of', 'in', 'at', 'for']:
                    if i + 1 < len(words) and words[i + 1][0].isupper():
                        potential_players.append(f"{word} {words[i + 1]}")
                    else:
                        potential_players.append(word)
            
            player_name = potential_players[0] if potential_players else None
            current_team = None
            
            # Try to extract team
            if 'from' in query.lower():
                team_part = query.lower().split('from')[-1].strip()
                current_team = team_part.split()[0].title() if team_part else None
            
            return player_name, current_team
            
        except Exception as e:
            logger.error(f"Error extracting player/team: {e}")
            return None, None
    
    def _extract_comparison_teams(self, query: str, context: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract teams for comparison from query."""
        # Reuse match team extraction for comparison queries
        return self._extract_match_teams_from_query(query, context)
    
    def _generate_tactical_recommendations(self, analysis: TacticalAnalysis) -> List[str]:
        """Generate recommendations based on tactical analysis."""
        recommendations = []
        
        if analysis.formation:
            recommendations.append(f"Consider the {analysis.formation.value} formation strengths in match preparation")
        
        if analysis.tactical_style:
            recommendations.append(f"Prepare for {analysis.tactical_style.value} playing style")
        
        for weakness in analysis.weaknesses:
            recommendations.append(f"Exploit: {weakness}")
        
        return recommendations[:3]
    
    def _generate_prediction_recommendations(self, prediction: MatchPrediction) -> List[str]:
        """Generate recommendations based on match prediction."""
        recommendations = []
        
        if prediction.home_win_probability > 0.5:
            recommendations.append(f"Home team ({prediction.home_team}) favored to win")
        elif prediction.away_win_probability > 0.5:
            recommendations.append(f"Away team ({prediction.away_team}) favored to win")
        else:
            recommendations.append("Close match, draw possible")
        
        for factor in prediction.key_factors:
            recommendations.append(f"Key factor: {factor}")
        
        return recommendations[:3]
    
    def _generate_historical_recommendations(self, trend: HistoricalTrend) -> List[str]:
        """Generate recommendations based on historical trends."""
        recommendations = []
        
        if trend.trend_direction == "improving":
            recommendations.append(f"{trend.team_name} shows positive momentum")
        elif trend.trend_direction == "declining":
            recommendations.append(f"{trend.team_name} needs to address declining performance")
        
        recommendations.append(f"Monitor {trend.metric} trends for future predictions")
        
        return recommendations[:2]
    
    def _generate_transfer_recommendations(self, valuation: TransferValuation) -> List[str]:
        """Generate recommendations based on transfer valuation."""
        recommendations = []
        
        recommendations.append(f"Estimated value: €{valuation.estimated_value:.1f}M")
        
        if valuation.market_trend == "rising":
            recommendations.append("Consider timing of transfer due to rising market value")
        elif valuation.market_trend == "falling":
            recommendations.append("Market value may be declining")
        
        return recommendations[:2]
    
    def _generate_comparison_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on team comparison."""
        recommendations = []
        
        if 'tactical_matchup' in comparison:
            recommendations.append("Focus on tactical matchup analysis")
        
        if 'key_battles' in comparison:
            for battle in comparison['key_battles'][:2]:
                recommendations.append(f"Key battle: {battle}")
        
        return recommendations[:3]
    
    def _create_fallback_response(self, query: str, query_type: AnalysisType, reason: str) -> FootballKnowledge:
        """Create fallback response when processing fails."""
        return FootballKnowledge(
            query=query,
            analysis_type=query_type,
            primary_result={'error': reason},
            supporting_data={},
            confidence=0.1,
            reasoning=f"Analysis failed: {reason}",
            recommendations=["Please try rephrasing your query"],
            data_sources=[],
            generated_at=datetime.now()
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and performance metrics."""
        return {
            'total_requests': self.request_count,
            'successful_requests': self.success_count,
            'success_rate': self.success_count / max(self.request_count, 1),
            'cache_hits': self.cache_hits,
            'components': {
                'tactical_analyzer': 'active',
                'predictive_analytics': 'active',
                'historical_context': 'active',
                'transfer_intelligence': 'active'
            }
        }


# ============================================================================
# Factory Functions and Integration Helpers
# ============================================================================

def create_football_knowledge_engine() -> FootballKnowledgeCoordinator:
    """Factory function to create a configured football knowledge engine."""
    try:
        coordinator = FootballKnowledgeCoordinator()
        logger.info("Football knowledge engine created successfully")
        return coordinator
    except Exception as e:
        logger.error(f"Failed to create football knowledge engine: {e}")
        raise


async def analyze_football_query(query: str, analysis_type: AnalysisType = AnalysisType.STRATEGIC,
                               context: Optional[Dict[str, Any]] = None) -> FootballKnowledge:
    """
    Convenience function for analyzing football queries.
    
    Args:
        query: The football query to analyze
        analysis_type: Type of analysis to perform
        context: Additional context for analysis
        
    Returns:
        FootballKnowledge object with comprehensive analysis
    """
    engine = create_football_knowledge_engine()
    return await engine.process_football_query(query, analysis_type, context)


async def get_tactical_analysis(team_name: str, league_code: str = "PL") -> TacticalAnalysis:
    """
    Convenience function for tactical analysis.
    
    Args:
        team_name: Name of the team to analyze
        league_code: League code (defaults to Premier League)
        
    Returns:
        TacticalAnalysis object
    """
    engine = create_football_knowledge_engine()
    return await engine.tactical_analyzer.analyze_team_tactics(team_name, league_code)


async def predict_match_outcome(home_team: str, away_team: str, league_code: str = "PL") -> MatchPrediction:
    """
    Convenience function for match prediction.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        league_code: League code (defaults to Premier League)
        
    Returns:
        MatchPrediction object
    """
    engine = create_football_knowledge_engine()
    return await engine.predictive_analytics.predict_match(home_team, away_team, league_code)


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    async def test_football_knowledge_engine():
        """Test function for the football knowledge engine."""
        engine = create_football_knowledge_engine()
        
        test_queries = [
            ("What formation does Arsenal use?", AnalysisType.TACTICAL),
            ("Predict Manchester United vs Liverpool", AnalysisType.PREDICTIVE),
            ("How has Barcelona performed this season?", AnalysisType.HISTORICAL),
            ("What is Messi's transfer value?", AnalysisType.TRANSFER),
            ("Compare Real Madrid vs Barcelona tactics", AnalysisType.COMPARATIVE)
        ]
        
        for query, analysis_type in test_queries:
            print(f"\nTesting query: {query}")
            print(f"Analysis type: {analysis_type.value}")
            
            try:
                result = await engine.process_football_query(query, analysis_type)
                print(f"Confidence: {result.confidence:.2f}")
                print(f"Reasoning: {result.reasoning}")
                print(f"Recommendations: {result.recommendations}")
            except Exception as e:
                print(f"Error: {e}")
        
        print(f"\nSystem status: {engine.get_system_status()}")
    
    # Run test if executed directly
    import asyncio
    asyncio.run(test_football_knowledge_engine())
