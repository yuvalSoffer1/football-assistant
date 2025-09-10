"""
Intelligent Entity Resolution System

This module implements LLM-driven entity resolution to replace hard-coded dictionaries
with intelligent understanding of team names, leagues, metrics, and other football entities.

Core Components:
- TeamResolver: Dynamic team name resolution using LLM context
- LeagueDetector: Context-aware league identification 
- MetricResolver: Intelligent football metric understanding
- EntityResolutionCoordinator: Main orchestrator for all entity resolution

Author: Football Assistant AI System
"""

import os
import re
import json
import time
import logging
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict

# Import existing modules for integration
from llm.openrouter_llm import query_llm
from api.football_data import get_standings, resolve_competition, get_teams
from api.api_football import get_live_scores

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Data Models and Enums
# ============================================================================

class EntityType(Enum):
    """Types of entities that can be resolved."""
    TEAM = "team"
    LEAGUE = "league" 
    PLAYER = "player"
    METRIC = "metric"
    TEMPORAL = "temporal"
    VENUE = "venue"
    COACH = "coach"
    UNKNOWN = "unknown"


class ConfidenceLevel(Enum):
    """Confidence levels for entity resolution."""
    VERY_HIGH = "very_high"  # > 0.95
    HIGH = "high"           # 0.85 - 0.95
    MEDIUM = "medium"       # 0.70 - 0.85
    LOW = "low"            # 0.50 - 0.70
    VERY_LOW = "very_low"  # < 0.50


@dataclass
class ResolvedEntity:
    """Represents a resolved entity with canonical information and metadata."""
    original_text: str
    entity_type: EntityType
    canonical_name: str
    entity_id: str
    confidence: float
    api_mappings: Dict[str, str]  # Map to different API identifiers
    aliases: List[str]
    metadata: Dict[str, Any]
    disambiguation_context: Optional[str] = None
    resolution_method: str = "llm"  # "llm", "fuzzy", "exact", "cache"
    
    def __post_init__(self):
        if self.api_mappings is None:
            self.api_mappings = {}
        if self.aliases is None:
            self.aliases = []
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get confidence level enum based on numeric confidence."""
        if self.confidence > 0.95:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence > 0.85:
            return ConfidenceLevel.HIGH
        elif self.confidence > 0.70:
            return ConfidenceLevel.MEDIUM
        elif self.confidence > 0.50:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


@dataclass
class ResolutionContext:
    """Context information to improve entity resolution accuracy."""
    query_text: str
    conversation_history: List[Dict[str, str]]
    current_league_focus: Optional[str] = None
    current_team_focus: Optional[str] = None
    temporal_context: Optional[str] = None
    user_preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.user_preferences is None:
            self.user_preferences = {}


@dataclass
class CacheEntry:
    """Cache entry for resolved entities."""
    entity: ResolvedEntity
    timestamp: datetime
    access_count: int = 0
    last_accessed: datetime = None
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.timestamp


# ============================================================================
# Team Resolver - Dynamic team name resolution using LLM context
# ============================================================================

class TeamResolver:
    """
    Intelligent team name resolution using LLM context and fuzzy matching.
    Replaces hard-coded team name dictionaries with dynamic understanding.
    """
    
    def __init__(self):
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_ttl_hours = 24
        self.api_cache: Dict[str, Dict] = {}  # Cache for API data
        self.known_teams: Dict[str, ResolvedEntity] = {}
        self.alias_mappings: Dict[str, str] = {}
        logger.info("TeamResolver initialized")
    
    async def resolve_team(self, team_text: str, context: ResolutionContext) -> Optional[ResolvedEntity]:
        """
        Resolve team name using intelligent LLM-driven approach.
        
        Args:
            team_text: The team name or reference to resolve
            context: Context information for better resolution
            
        Returns:
            ResolvedEntity if successful, None otherwise
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(team_text, context)
            cached = self._get_cached_team(cache_key)
            if cached:
                logger.info(f"Cache hit for team: {team_text}")
                return cached
            
            # Try multiple resolution methods in order of confidence
            resolution_methods = [
                self._resolve_via_exact_match,
                self._resolve_via_api_football,
                self._resolve_via_football_data,
                self._resolve_via_llm_understanding,
                self._resolve_via_fuzzy_matching
            ]
            
            for method in resolution_methods:
                try:
                    result = await method(team_text, context)
                    if result and result.confidence > 0.5:  # Minimum confidence threshold
                        self._cache_team(cache_key, result)
                        return result
                except Exception as e:
                    logger.warning(f"Team resolution method {method.__name__} failed: {e}")
                    continue
            
            logger.warning(f"Could not resolve team: {team_text}")
            return None
            
        except Exception as e:
            logger.error(f"Error resolving team '{team_text}': {str(e)}")
            return None
    
    async def _resolve_via_exact_match(self, team_text: str, context: ResolutionContext) -> Optional[ResolvedEntity]:
        """Try exact match against known teams."""
        normalized = self._normalize_team_name(team_text)
        
        if normalized in self.known_teams:
            entity = self.known_teams[normalized]
            entity.resolution_method = "exact"
            entity.confidence = 0.98
            return entity
        
        return None
    
    async def _resolve_via_api_football(self, team_text: str, context: ResolutionContext) -> Optional[ResolvedEntity]:
        """Resolve team using API-Football search."""
        try:
            api_key = os.getenv("API_FOOTBALL_KEY")
            if not api_key:
                return None
            
            # Use existing API-Football logic from main.py
            from main import _apif_find_team_and_code
            result = _apif_find_team_and_code(team_text)
            
            if result:
                team_name, fd_code = result
                return ResolvedEntity(
                    original_text=team_text,
                    entity_type=EntityType.TEAM,
                    canonical_name=team_name,
                    entity_id=self._generate_team_id(team_name),
                    confidence=0.90,
                    api_mappings={
                        'football_data': fd_code,
                        'api_football': team_name
                    },
                    aliases=[team_text, team_name],
                    metadata={'source': 'api_football'},
                    resolution_method="api_football"
                )
        except Exception as e:
            logger.warning(f"API-Football team resolution failed: {e}")
        
        return None
    
    async def _resolve_via_football_data(self, team_text: str, context: ResolutionContext) -> Optional[ResolvedEntity]:
        """Resolve team by searching football-data league tables."""
        try:
            # Use existing football-data probe logic from main.py
            from main import _fd_probe_for_team
            result = _fd_probe_for_team(team_text)
            
            if result:
                team_name, fd_code = result
                return ResolvedEntity(
                    original_text=team_text,
                    entity_type=EntityType.TEAM,
                    canonical_name=team_name,
                    entity_id=self._generate_team_id(team_name),
                    confidence=0.85,
                    api_mappings={'football_data': fd_code},
                    aliases=[team_text, team_name],
                    metadata={'source': 'football_data'},
                    resolution_method="football_data"
                )
        except Exception as e:
            logger.warning(f"Football-Data team resolution failed: {e}")
        
        return None
    
    async def _resolve_via_llm_understanding(self, team_text: str, context: ResolutionContext) -> Optional[ResolvedEntity]:
        """Use LLM to understand and resolve team names with context."""
        try:
            # Build context-aware prompt
            context_info = ""
            if context.current_league_focus:
                context_info += f"Current league context: {context.current_league_focus}\n"
            if context.conversation_history:
                recent_history = context.conversation_history[-3:]
                context_info += f"Recent conversation: {recent_history}\n"
            
            llm_prompt = f"""
            You are a football team name resolver. Analyze the text and identify the most likely team.
            
            Text to resolve: "{team_text}"
            Context: {context_info}
            
            Consider:
            - Common abbreviations (e.g., "Man U" = "Manchester United", "Barca" = "FC Barcelona")
            - Nicknames (e.g., "The Reds", "Los Blancos")
            - Historical names and rebrands
            - Multi-language variations
            - Context clues from conversation
            
            Return JSON:
            {{
                "canonical_name": "Official team name",
                "confidence": 0.95,
                "reasoning": "Why this is the correct team",
                "league": "Most likely current league",
                "country": "Team's country",
                "aliases": ["alternative", "names"]
            }}
            
            If uncertain, return confidence < 0.7. If no valid team found, return {{"canonical_name": null}}.
            """
            
            response = query_llm(
                question=llm_prompt,
                scores={},
                standings={},
                history=context.conversation_history,
                focus=context.current_league_focus or ""
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                result = json.loads(json_match.group())
                
                if result.get('canonical_name'):
                    return ResolvedEntity(
                        original_text=team_text,
                        entity_type=EntityType.TEAM,
                        canonical_name=result['canonical_name'],
                        entity_id=self._generate_team_id(result['canonical_name']),
                        confidence=result.get('confidence', 0.8),
                        api_mappings={},
                        aliases=result.get('aliases', [team_text]),
                        metadata={
                            'source': 'llm',
                            'reasoning': result.get('reasoning', ''),
                            'league': result.get('league', ''),
                            'country': result.get('country', '')
                        },
                        resolution_method="llm"
                    )
        except Exception as e:
            logger.warning(f"LLM team resolution failed: {e}")
        
        return None
    
    async def _resolve_via_fuzzy_matching(self, team_text: str, context: ResolutionContext) -> Optional[ResolvedEntity]:
        """Fallback fuzzy matching against known team names."""
        try:
            import difflib
            
            # Get all known team names from cache and API sources
            candidates = list(self.known_teams.keys())
            
            # Add teams from current league context if available
            if context.current_league_focus:
                try:
                    league_teams = await self._get_league_teams(context.current_league_focus)
                    candidates.extend(league_teams)
                except Exception:
                    pass
            
            if not candidates:
                return None
            
            # Normalize and find best matches
            normalized_text = self._normalize_team_name(team_text)
            normalized_candidates = [self._normalize_team_name(c) for c in candidates]
            
            best_matches = difflib.get_close_matches(
                normalized_text, 
                normalized_candidates, 
                n=1, 
                cutoff=0.6
            )
            
            if best_matches:
                best_match = best_matches[0]
                original_candidate = candidates[normalized_candidates.index(best_match)]
                
                confidence = difflib.SequenceMatcher(None, normalized_text, best_match).ratio()
                
                return ResolvedEntity(
                    original_text=team_text,
                    entity_type=EntityType.TEAM,
                    canonical_name=original_candidate,
                    entity_id=self._generate_team_id(original_candidate),
                    confidence=confidence * 0.8,  # Reduce confidence for fuzzy matches
                    api_mappings={},
                    aliases=[team_text, original_candidate],
                    metadata={'source': 'fuzzy_match'},
                    resolution_method="fuzzy"
                )
        except Exception as e:
            logger.warning(f"Fuzzy team matching failed: {e}")
        
        return None
    
    def _normalize_team_name(self, name: str) -> str:
        """Normalize team name for comparison."""
        name = (name or "").lower().strip()
        # Remove common prefixes/suffixes
        name = re.sub(r'\b(fc|cf|ac|rc|sc|ud|cd|rcd|real|club)\b', '', name)
        # Remove punctuation and extra spaces
        name = re.sub(r'[^\w\s]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        return name
    
    def _generate_team_id(self, team_name: str) -> str:
        """Generate a consistent team ID."""
        normalized = self._normalize_team_name(team_name)
        return re.sub(r'[^\w]', '_', normalized).lower()
    
    def _generate_cache_key(self, team_text: str, context: ResolutionContext) -> str:
        """Generate cache key for team resolution."""
        context_str = f"{context.current_league_focus or ''}"
        return hashlib.md5(f"{team_text.lower()}|{context_str}".encode()).hexdigest()
    
    def _get_cached_team(self, cache_key: str) -> Optional[ResolvedEntity]:
        """Get cached team resolution."""
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if datetime.now() - entry.timestamp < timedelta(hours=self.cache_ttl_hours):
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                return entry.entity
            else:
                del self.cache[cache_key]
        return None
    
    def _cache_team(self, cache_key: str, entity: ResolvedEntity):
        """Cache team resolution."""
        self.cache[cache_key] = CacheEntry(
            entity=entity,
            timestamp=datetime.now(),
            access_count=1,
            last_accessed=datetime.now()
        )
    
    async def _get_league_teams(self, league_code: str) -> List[str]:
        """Get team names from a specific league."""
        try:
            if league_code in self.api_cache:
                return self.api_cache[league_code]
            
            standings = get_standings(league_code)
            teams = []
            
            for standing in standings.get('standings', []):
                for team_row in standing.get('table', []):
                    team_name = team_row.get('team', {}).get('name', '')
                    if team_name:
                        teams.append(team_name)
            
            self.api_cache[league_code] = teams
            return teams
        except Exception:
            return []


# ============================================================================
# League Detector - Context-aware league identification
# ============================================================================

class LeagueDetector:
    """
    Intelligent league detection using LLM context and multi-source resolution.
    Replaces hard-coded league mapping dictionaries.
    """
    
    def __init__(self):
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_ttl_hours = 12
        self.league_aliases: Dict[str, str] = {}
        self.competition_data: Optional[Dict] = None
        logger.info("LeagueDetector initialized")
    
    async def detect_league(self, text: str, context: ResolutionContext) -> Optional[ResolvedEntity]:
        """
        Detect league from text using intelligent methods.
        
        Args:
            text: Text containing potential league references
            context: Context for better detection
            
        Returns:
            ResolvedEntity for the detected league
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(text, context)
            cached = self._get_cached_league(cache_key)
            if cached:
                logger.info(f"Cache hit for league: {text[:50]}")
                return cached
            
            # Try resolution methods in order
            methods = [
                self._detect_via_exact_match,
                self._detect_via_existing_logic,
                self._detect_via_llm_understanding,
                self._detect_via_fuzzy_matching
            ]
            
            for method in methods:
                try:
                    result = await method(text, context)
                    if result and result.confidence > 0.6:
                        self._cache_league(cache_key, result)
                        return result
                except Exception as e:
                    logger.warning(f"League detection method {method.__name__} failed: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting league from '{text}': {str(e)}")
            return None
    
    async def _detect_via_exact_match(self, text: str, context: ResolutionContext) -> Optional[ResolvedEntity]:
        """Try exact match against known league codes and names."""
        text_lower = text.lower().strip()
        
        # Check direct code matches
        known_codes = ['PL', 'PD', 'SA', 'BL1', 'FL1', 'DED', 'PPL', 'CL', 'ELC', 'SPL']
        for code in known_codes:
            if code.lower() in text_lower:
                return await self._create_league_entity(text, code, 0.95, "exact_code")
        
        return None
    
    async def _detect_via_existing_logic(self, text: str, context: ResolutionContext) -> Optional[ResolvedEntity]:
        """Use existing league detection logic."""
        try:
            # Use existing resolve_competition from football_data.py
            code = resolve_competition(text)
            if code:
                return await self._create_league_entity(text, code, 0.90, "existing_logic")
        except Exception as e:
            logger.warning(f"Existing league logic failed: {e}")
        
        return None
    
    async def _detect_via_llm_understanding(self, text: str, context: ResolutionContext) -> Optional[ResolvedEntity]:
        """Use LLM for intelligent league understanding."""
        try:
            context_info = ""
            if context.conversation_history:
                context_info = f"Recent conversation: {context.conversation_history[-2:]}"
            
            llm_prompt = f"""
            You are a football league detector. Analyze the text and identify any football league mentioned.
            
            Text: "{text}"
            Context: {context_info}
            
            Consider:
            - Official league names in multiple languages
            - Common abbreviations and nicknames
            - Branding variations (e.g., "LaLiga EA Sports", "Premier League")
            - Historical names
            - Context clues
            
            Known league codes:
            - PL: Premier League (England)
            - PD: La Liga (Spain) 
            - SA: Serie A (Italy)
            - BL1: Bundesliga (Germany)
            - FL1: Ligue 1 (France)
            - CL: Champions League
            - DED: Eredivisie (Netherlands)
            - PPL: Primeira Liga (Portugal)
            
            Return JSON:
            {{
                "league_code": "PL",
                "league_name": "Premier League", 
                "confidence": 0.95,
                "reasoning": "Why this league was detected"
            }}
            
            If no league found, return {{"league_code": null}}.
            """
            
            response = query_llm(
                question=llm_prompt,
                scores={},
                standings={},
                history=context.conversation_history,
                focus=""
            )
            
            # Extract JSON
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                result = json.loads(json_match.group())
                
                if result.get('league_code'):
                    return ResolvedEntity(
                        original_text=text,
                        entity_type=EntityType.LEAGUE,
                        canonical_name=result.get('league_name', result['league_code']),
                        entity_id=result['league_code'],
                        confidence=result.get('confidence', 0.8),
                        api_mappings={'football_data': result['league_code']},
                        aliases=[text],
                        metadata={
                            'source': 'llm',
                            'reasoning': result.get('reasoning', '')
                        },
                        resolution_method="llm"
                    )
        except Exception as e:
            logger.warning(f"LLM league detection failed: {e}")
        
        return None
    
    async def _detect_via_fuzzy_matching(self, text: str, context: ResolutionContext) -> Optional[ResolvedEntity]:
        """Fuzzy match against known league names."""
        try:
            import difflib
            
            league_names = {
                'premier league': 'PL',
                'english premier league': 'PL',
                'la liga': 'PD', 
                'laliga': 'PD',
                'primera division': 'PD',
                'serie a': 'SA',
                'bundesliga': 'BL1',
                'ligue 1': 'FL1',
                'champions league': 'CL',
                'eredivisie': 'DED',
                'primeira liga': 'PPL'
            }
            
            text_lower = text.lower()
            candidates = list(league_names.keys())
            
            best_matches = difflib.get_close_matches(text_lower, candidates, n=1, cutoff=0.7)
            
            if best_matches:
                match = best_matches[0]
                code = league_names[match]
                confidence = difflib.SequenceMatcher(None, text_lower, match).ratio()
                
                return await self._create_league_entity(text, code, confidence * 0.8, "fuzzy")
        except Exception as e:
            logger.warning(f"Fuzzy league matching failed: {e}")
        
        return None
    
    async def _create_league_entity(self, original_text: str, league_code: str, 
                                   confidence: float, method: str) -> ResolvedEntity:
        """Create a league entity from code."""
        league_names = {
            'PL': 'Premier League',
            'PD': 'La Liga', 
            'SA': 'Serie A',
            'BL1': 'Bundesliga',
            'FL1': 'Ligue 1',
            'CL': 'Champions League',
            'DED': 'Eredivisie',
            'PPL': 'Primeira Liga',
            'ELC': 'EFL Championship',
            'SPL': 'Scottish Premiership'
        }
        
        return ResolvedEntity(
            original_text=original_text,
            entity_type=EntityType.LEAGUE,
            canonical_name=league_names.get(league_code, league_code),
            entity_id=league_code,
            confidence=confidence,
            api_mappings={'football_data': league_code},
            aliases=[original_text],
            metadata={'source': method},
            resolution_method=method
        )
    
    def _generate_cache_key(self, text: str, context: ResolutionContext) -> str:
        """Generate cache key for league detection."""
        return hashlib.md5(f"league:{text.lower()}".encode()).hexdigest()
    
    def _get_cached_league(self, cache_key: str) -> Optional[ResolvedEntity]:
        """Get cached league detection."""
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if datetime.now() - entry.timestamp < timedelta(hours=self.cache_ttl_hours):
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                return entry.entity
            else:
                del self.cache[cache_key]
        return None
    
    def _cache_league(self, cache_key: str, entity: ResolvedEntity):
        """Cache league detection."""
        self.cache[cache_key] = CacheEntry(
            entity=entity,
            timestamp=datetime.now(),
            access_count=1,
            last_accessed=datetime.now()
        )


# ============================================================================
# Metric Resolver - Intelligent football metric understanding
# ============================================================================

class MetricResolver:
    """
    Intelligent football metric resolution using LLM understanding.
    Replaces hard-coded metric aliases with dynamic understanding.
    """
    
    def __init__(self):
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_ttl_hours = 6
        self.metric_ontology: Dict[str, Dict] = self._build_metric_ontology()
        logger.info("MetricResolver initialized")
    
    async def resolve_metric(self, text: str, context: ResolutionContext) -> Optional[ResolvedEntity]:
        """
        Resolve football metrics from text using intelligent understanding.
        
        Args:
            text: Text containing metric references
            context: Context for better resolution
            
        Returns:
            ResolvedEntity for the detected metric
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(text, context)
            cached = self._get_cached_metric(cache_key)
            if cached:
                return cached
            
            # Try resolution methods
            methods = [
                self._resolve_via_direct_mapping,
                self._resolve_via_llm_understanding,
                self._resolve_via_contextual_inference
            ]
            
            for method in methods:
                try:
                    result = await method(text, context)
                    if result and result.confidence > 0.6:
                        self._cache_metric(cache_key, result)
                        return result
                except Exception as e:
                    logger.warning(f"Metric resolution method {method.__name__} failed: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error resolving metric from '{text}': {str(e)}")
            return None
    
    async def _resolve_via_direct_mapping(self, text: str, context: ResolutionContext) -> Optional[ResolvedEntity]:
        """Try direct mapping against known metric patterns."""
        text_lower = text.lower()
        
        for canonical_metric, data in self.metric_ontology.items():
            for alias in data['aliases']:
                if alias in text_lower:
                    return ResolvedEntity(
                        original_text=text,
                        entity_type=EntityType.METRIC,
                        canonical_name=data['display_name'],
                        entity_id=canonical_metric,
                        confidence=0.95,
                        api_mappings={'football_data': canonical_metric},
                        aliases=data['aliases'],
                        metadata={
                            'source': 'direct_mapping',
                            'category': data['category'],
                            'description': data['description']
                        },
                        resolution_method="direct"
                    )
        
        return None
    
    async def _resolve_via_llm_understanding(self, text: str, context: ResolutionContext) -> Optional[ResolvedEntity]:
        """Use LLM for intelligent metric understanding."""
        try:
            metrics_info = self._get_metrics_description()
            
            llm_prompt = f"""
            You are a football statistics expert. Analyze the text and identify any football metric being requested.
            
            Text: "{text}"
            
            Available metrics:
            {metrics_info}
            
            Consider:
            - Synonyms and variations (e.g., "scored" = "goals for", "conceded" = "goals against")
            - Abbreviations (e.g., "GD" = "goal difference", "GF" = "goals for")
            - Context clues about what statistic is being asked for
            - Multi-language terms
            
            Return JSON:
            {{
                "metric_id": "goalDifference",
                "metric_name": "Goal Difference",
                "confidence": 0.95,
                "reasoning": "Why this metric was detected"
            }}
            
            If no clear metric found, return {{"metric_id": null}}.
            """
            
            response = query_llm(
                question=llm_prompt,
                scores={},
                standings={},
                history=context.conversation_history,
                focus=""
            )
            
            # Extract JSON
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                result = json.loads(json_match.group())
                
                if result.get('metric_id') and result['metric_id'] in self.metric_ontology:
                    metric_data = self.metric_ontology[result['metric_id']]
                    
                    return ResolvedEntity(
                        original_text=text,
                        entity_type=EntityType.METRIC,
                        canonical_name=result.get('metric_name', metric_data['display_name']),
                        entity_id=result['metric_id'],
                        confidence=result.get('confidence', 0.8),
                        api_mappings={'football_data': result['metric_id']},
                        aliases=metric_data['aliases'],
                        metadata={
                            'source': 'llm',
                            'reasoning': result.get('reasoning', ''),
                            'category': metric_data['category']
                        },
                        resolution_method="llm"
                    )
        except Exception as e:
            logger.warning(f"LLM metric resolution failed: {e}")
        
        return None
    
    async def _resolve_via_contextual_inference(self, text: str, context: ResolutionContext) -> Optional[ResolvedEntity]:
        """Infer metric from context and question structure."""
        text_lower = text.lower()
        
        # Pattern-based inference
        patterns = {
            r'how many.*goals.*scored': 'goalsFor',
            r'how many.*goals.*conceded': 'goalsAgainst', 
            r'what.*position': 'position',
            r'how many.*points': 'points',
            r'how many.*wins': 'won',
            r'goal.*difference': 'goalDifference',
            r'table.*position': 'position'
        }
        
        for pattern, metric_id in patterns.items():
            if re.search(pattern, text_lower):
                if metric_id in self.metric_ontology:
                    metric_data = self.metric_ontology[metric_id]
                    
                    return ResolvedEntity(
                        original_text=text,
                        entity_type=EntityType.METRIC,
                        canonical_name=metric_data['display_name'],
                        entity_id=metric_id,
                        confidence=0.75,
                        api_mappings={'football_data': metric_id},
                        aliases=metric_data['aliases'],
                        metadata={
                            'source': 'contextual_inference',
                            'pattern': pattern,
                            'category': metric_data['category']
                        },
                        resolution_method="contextual"
                    )
        
        return None
    
    def _build_metric_ontology(self) -> Dict[str, Dict]:
        """Build comprehensive metric ontology."""
        return {
            "points": {
                "display_name": "Points",
                "aliases": ["points", "pts", "point", "league points"],
                "category": "standings",
                "description": "Total points earned in the league"
            },
            "position": {
                "display_name": "Position",
                "aliases": ["position", "rank", "place", "table position", "standing"],
                "category": "standings", 
                "description": "Current position in the league table"
            },
            "goalDifference": {
                "display_name": "Goal Difference",
                "aliases": ["gd", "goal difference", "goal diff", "goal-difference", "+/-"],
                "category": "goals",
                "description": "Difference between goals scored and conceded"
            },
            "goalsFor": {
                "display_name": "Goals For",
                "aliases": ["gf", "goals for", "goals scored", "scored", "for"],
                "category": "goals",
                "description": "Total goals scored by the team"
            },
            "goalsAgainst": {
                "display_name": "Goals Against", 
                "aliases": ["ga", "goals against", "goals conceded", "conceded", "against"],
                "category": "goals",
                "description": "Total goals conceded by the team"
            },
            "won": {
                "display_name": "Wins",
                "aliases": ["won", "wins", "win", "w", "victories"],
                "category": "results",
                "description": "Number of matches won"
            },
            "draw": {
                "display_name": "Draws",
                "aliases": ["draw", "draws", "tied", "tie", "d"],
                "category": "results", 
                "description": "Number of matches drawn"
            },
            "lost": {
                "display_name": "Losses",
                "aliases": ["lost", "loss", "losses", "defeats", "l"],
                "category": "results",
                "description": "Number of matches lost"
            },
            "playedGames": {
                "display_name": "Played",
                "aliases": ["played", "games", "matches", "played games", "mp", "p"],
                "category": "general",
                "description": "Number of matches played"
            }
        }
    
    def _get_metrics_description(self) -> str:
        """Get formatted description of available metrics."""
        lines = []
        for metric_id, data in self.metric_ontology.items():
            aliases_str = ", ".join(data['aliases'][:5])  # Limit for brevity
            lines.append(f"- {metric_id}: {data['display_name']} ({aliases_str})")
        return "\n".join(lines)
    
    def _generate_cache_key(self, text: str, context: ResolutionContext) -> str:
        """Generate cache key for metric resolution."""
        return hashlib.md5(f"metric:{text.lower()}".encode()).hexdigest()
    
    def _get_cached_metric(self, cache_key: str) -> Optional[ResolvedEntity]:
        """Get cached metric resolution."""
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if datetime.now() - entry.timestamp < timedelta(hours=self.cache_ttl_hours):
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                return entry.entity
            else:
                del self.cache[cache_key]
        return None
    
    def _cache_metric(self, cache_key: str, entity: ResolvedEntity):
        """Cache metric resolution."""
        self.cache[cache_key] = CacheEntry(
            entity=entity,
            timestamp=datetime.now(),
            access_count=1,
            last_accessed=datetime.now()
        )


# ============================================================================
# Entity Resolution Coordinator - Main orchestrator
# ============================================================================

class EntityResolutionCoordinator:
    """
    Main orchestrator for all entity resolution operations.
    Coordinates between different resolvers and provides unified interface.
    """
    
    def __init__(self):
        self.team_resolver = TeamResolver()
        self.league_detector = LeagueDetector()
        self.metric_resolver = MetricResolver()
        
        # Global cache for resolved entities
        self.global_cache: Dict[str, Dict[EntityType, List[ResolvedEntity]]] = defaultdict(lambda: defaultdict(list))
        self.cache_ttl_hours = 24
        
        # Performance metrics
        self.resolution_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'successful_resolutions': 0,
            'failed_resolutions': 0,
            'avg_resolution_time': 0.0
        }
        
        logger.info("EntityResolutionCoordinator initialized")
    
    async def resolve_entities(self, text: str, context: ResolutionContext, 
                             entity_types: Optional[List[EntityType]] = None) -> List[ResolvedEntity]:
        """
        Resolve all entities in text using intelligent coordination.
        
        Args:
            text: Text to analyze for entities
            context: Context for better resolution
            entity_types: Specific entity types to look for (None = all types)
            
        Returns:
            List of resolved entities
        """
        start_time = time.time()
        self.resolution_stats['total_requests'] += 1
        
        try:
            # Default to all entity types if not specified
            if entity_types is None:
                entity_types = [EntityType.TEAM, EntityType.LEAGUE, EntityType.METRIC]
            
            resolved_entities = []
            
            # Check global cache first
            cache_key = self._generate_global_cache_key(text, context)
            cached_results = self._get_global_cache(cache_key, entity_types)
            if cached_results:
                self.resolution_stats['cache_hits'] += 1
                logger.info(f"Global cache hit for: {text[:50]}")
                return cached_results
            
            # Resolve each entity type
            tasks = []
            if EntityType.TEAM in entity_types:
                tasks.append(self._resolve_teams(text, context))
            if EntityType.LEAGUE in entity_types:
                tasks.append(self._resolve_leagues(text, context))
            if EntityType.METRIC in entity_types:
                tasks.append(self._resolve_metrics(text, context))
            
            # Execute resolution tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect successful results
            for result in results:
                if isinstance(result, list):
                    resolved_entities.extend(result)
                elif isinstance(result, ResolvedEntity):
                    resolved_entities.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Entity resolution task failed: {result}")
            
            # Post-process and validate results
            resolved_entities = self._post_process_entities(resolved_entities, text, context)
            
            # Cache results
            self._cache_global_results(cache_key, resolved_entities)
            
            # Update stats
            if resolved_entities:
                self.resolution_stats['successful_resolutions'] += 1
            else:
                self.resolution_stats['failed_resolutions'] += 1
            
            resolution_time = time.time() - start_time
            self._update_avg_resolution_time(resolution_time)
            
            logger.info(f"Resolved {len(resolved_entities)} entities in {resolution_time:.3f}s")
            return resolved_entities
            
        except Exception as e:
            logger.error(f"Error in entity resolution coordination: {str(e)}")
            self.resolution_stats['failed_resolutions'] += 1
            return []
    
    async def resolve_single_entity(self, text: str, entity_type: EntityType, 
                                   context: ResolutionContext) -> Optional[ResolvedEntity]:
        """
        Resolve a single entity of specific type.
        
        Args:
            text: Text containing the entity
            entity_type: Type of entity to resolve
            context: Context for resolution
            
        Returns:
            ResolvedEntity if found, None otherwise
        """
        try:
            if entity_type == EntityType.TEAM:
                return await self.team_resolver.resolve_team(text, context)
            elif entity_type == EntityType.LEAGUE:
                return await self.league_detector.detect_league(text, context)
            elif entity_type == EntityType.METRIC:
                return await self.metric_resolver.resolve_metric(text, context)
            else:
                logger.warning(f"Unsupported entity type: {entity_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error resolving single entity '{text}' of type {entity_type}: {str(e)}")
            return None
    
    async def _resolve_teams(self, text: str, context: ResolutionContext) -> List[ResolvedEntity]:
        """Resolve team entities from text."""
        try:
            # Extract potential team mentions using multiple strategies
            team_candidates = self._extract_team_candidates(text)
            
            teams = []
            for candidate in team_candidates:
                team_entity = await self.team_resolver.resolve_team(candidate, context)
                if team_entity:
                    teams.append(team_entity)
            
            return teams
        except Exception as e:
            logger.warning(f"Team resolution failed: {e}")
            return []
    
    async def _resolve_leagues(self, text: str, context: ResolutionContext) -> List[ResolvedEntity]:
        """Resolve league entities from text."""
        try:
            league_entity = await self.league_detector.detect_league(text, context)
            return [league_entity] if league_entity else []
        except Exception as e:
            logger.warning(f"League resolution failed: {e}")
            return []
    
    async def _resolve_metrics(self, text: str, context: ResolutionContext) -> List[ResolvedEntity]:
        """Resolve metric entities from text."""
        try:
            metric_entity = await self.metric_resolver.resolve_metric(text, context)
            return [metric_entity] if metric_entity else []
        except Exception as e:
            logger.warning(f"Metric resolution failed: {e}")
            return []
    
    def _extract_team_candidates(self, text: str) -> List[str]:
        """Extract potential team name candidates from text."""
        candidates = []
        
        # Strategy 1: Look for common team name patterns
        team_patterns = [
            r'\b(?:fc|cf|ac|rc|sc)\s+\w+',  # FC Barcelona, etc.
            r'\w+\s+(?:fc|cf|ac|rc|sc)\b',  # Barcelona FC, etc.
            r'\b(?:real|atletico|athletic)\s+\w+',  # Real Madrid, etc.
            r'\b\w+\s+(?:united|city|town|rovers|wanderers|albion)\b',  # Manchester United, etc.
            r'\b(?:manchester|liverpool|arsenal|chelsea|tottenham|everton)\b',  # Premier League teams
            r'\b(?:barcelona|madrid|sevilla|valencia|bilbao)\b',  # La Liga teams
            r'\b(?:juventus|milan|napoli|roma|lazio|inter)\b',  # Serie A teams
            r'\b(?:bayern|dortmund|leipzig|leverkusen)\b',  # Bundesliga teams
        ]
        
        for pattern in team_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                candidate = match.group().strip()
                if candidate and len(candidate) > 2:
                    candidates.append(candidate)
        
        # Strategy 2: Extract proper nouns that might be teams
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for noun in proper_nouns:
            if len(noun) > 3 and noun not in ['Premier', 'League', 'Serie', 'Liga', 'Champions']:
                candidates.append(noun)
        
        # Remove duplicates and return
        return list(dict.fromkeys(candidates))  # Preserves order while removing duplicates
    
    def _post_process_entities(self, entities: List[ResolvedEntity], text: str, 
                              context: ResolutionContext) -> List[ResolvedEntity]:
        """Post-process resolved entities for consistency and validation."""
        # Remove duplicates based on entity_id
        seen_ids = set()
        unique_entities = []
        
        for entity in entities:
            if entity.entity_id not in seen_ids:
                seen_ids.add(entity.entity_id)
                unique_entities.append(entity)
        
        # Sort by confidence (highest first)
        unique_entities.sort(key=lambda e: e.confidence, reverse=True)
        
        # Filter out very low confidence results
        filtered_entities = [e for e in unique_entities if e.confidence >= 0.5]
        
        return filtered_entities
    
    def _generate_global_cache_key(self, text: str, context: ResolutionContext) -> str:
        """Generate cache key for global entity resolution."""
        context_str = f"{context.current_league_focus or ''}|{context.current_team_focus or ''}"
        return hashlib.md5(f"global:{text.lower()}|{context_str}".encode()).hexdigest()
    
    def _get_global_cache(self, cache_key: str, entity_types: List[EntityType]) -> Optional[List[ResolvedEntity]]:
        """Get cached global resolution results."""
        if cache_key in self.global_cache:
            cached_entities = []
            for entity_type in entity_types:
                if entity_type in self.global_cache[cache_key]:
                    cached_entities.extend(self.global_cache[cache_key][entity_type])
            
            # Check if cache is still valid
            if cached_entities and all(
                datetime.now() - entity.metadata.get('cached_at', datetime.min) < timedelta(hours=self.cache_ttl_hours)
                for entity in cached_entities
            ):
                return cached_entities
            else:
                # Remove expired cache
                del self.global_cache[cache_key]
        
        return None
    
    def _cache_global_results(self, cache_key: str, entities: List[ResolvedEntity]):
        """Cache global resolution results."""
        # Add timestamp to metadata
        for entity in entities:
            entity.metadata['cached_at'] = datetime.now()
        
        # Group by entity type
        for entity in entities:
            self.global_cache[cache_key][entity.entity_type].append(entity)
    
    def _update_avg_resolution_time(self, resolution_time: float):
        """Update average resolution time statistic."""
        current_avg = self.resolution_stats['avg_resolution_time']
        total_requests = self.resolution_stats['total_requests']
        
        # Calculate new average
        new_avg = ((current_avg * (total_requests - 1)) + resolution_time) / total_requests
        self.resolution_stats['avg_resolution_time'] = new_avg
    
    def get_resolution_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring."""
        return {
            **self.resolution_stats,
            'cache_hit_rate': (
                self.resolution_stats['cache_hits'] / max(1, self.resolution_stats['total_requests'])
            ),
            'success_rate': (
                self.resolution_stats['successful_resolutions'] / max(1, self.resolution_stats['total_requests'])
            ),
            'team_cache_size': len(self.team_resolver.cache),
            'league_cache_size': len(self.league_detector.cache),
            'metric_cache_size': len(self.metric_resolver.cache),
            'global_cache_size': len(self.global_cache)
        }
    
    def clear_caches(self):
        """Clear all caches for memory management."""
        self.team_resolver.cache.clear()
        self.league_detector.cache.clear() 
        self.metric_resolver.cache.clear()
        self.global_cache.clear()
        logger.info("All caches cleared")


# ============================================================================
# Factory Functions and Integration Helpers
# ============================================================================

def create_entity_resolver() -> EntityResolutionCoordinator:
    """Factory function to create a configured entity resolution coordinator."""
    try:
        coordinator = EntityResolutionCoordinator()
        logger.info("Entity resolution coordinator created successfully")
        return coordinator
    except Exception as e:
        logger.error(f"Failed to create entity resolution coordinator: {e}")
        raise


async def resolve_entities_from_query(query: str, conversation_history: List[Dict[str, str]] = None,
                                     current_focus: Dict[str, str] = None) -> List[ResolvedEntity]:
    """
    Convenience function for resolving entities from a query.
    
    Args:
        query: The user query to analyze
        conversation_history: Recent conversation history
        current_focus: Current focus context (league, team, etc.)
        
    Returns:
        List of resolved entities
    """
    coordinator = create_entity_resolver()
    
    context = ResolutionContext(
        query_text=query,
        conversation_history=conversation_history or [],
        current_league_focus=current_focus.get('league') if current_focus else None,
        current_team_focus=current_focus.get('team') if current_focus else None
    )
    
    return await coordinator.resolve_entities(query, context)


# Example usage and testing
if __name__ == "__main__":
    async def test_entity_resolver():
        """Test function for the entity resolution system."""
        coordinator = create_entity_resolver()
        
        test_queries = [
            "What's Manchester United's goal difference in the Premier League?",
            "How many points does Barcelona have in La Liga?", 
            "Show me Arsenal's position in the table",
            "Real Madrid vs Liverpool - who has more wins?",
            "Bayern Munich goals scored this season"
        ]
        
        for query in test_queries:
            print(f"\nTesting query: {query}")
            
            context = ResolutionContext(
                query_text=query,
                conversation_history=[],
                current_league_focus=None
            )
            
            entities = await coordinator.resolve_entities(query, context)
            
            print(f"Found {len(entities)} entities:")
            for entity in entities:
                print(f"  - {entity.entity_type.value}: {entity.original_text} -> {entity.canonical_name}")
                print(f"    Confidence: {entity.confidence:.2f}, Method: {entity.resolution_method}")
        
        print(f"\nResolution stats: {coordinator.get_resolution_stats()}")
    
    # Run test if executed directly
    import asyncio
    asyncio.run(test_entity_resolver())