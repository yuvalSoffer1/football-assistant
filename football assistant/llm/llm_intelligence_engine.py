"""
LLM Intelligence Engine - Core component for LLM-driven football assistant functionality.

This module implements the central intelligence layer that coordinates all LLM-driven operations,
replacing the dictionary-based processing in the current system with sophisticated natural
language understanding and context-aware responses.

Components:
- QueryProcessor: Natural language understanding and intent classification
- ContextManager: Conversation memory and context tracking
- IntelligenceCoordinator: Main orchestrator for LLM-driven operations
"""

import os
import re
import json
import time
import logging
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum

# Import existing modules for integration
from llm.openrouter_llm import query_llm
from api.football_data import get_standings, resolve_competition
from api.api_football import get_live_scores
from api.livescore_api import get_live_scores_lsa

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Data Models and Enums
# ============================================================================

class QueryType(Enum):
    """Classification of different query types for football questions."""
    STATISTICAL = "statistical"          # "How many goals has Messi scored?"
    COMPARATIVE = "comparative"          # "Who is better, Ronaldo or Messi?"
    PREDICTIVE = "predictive"           # "Who will win tomorrow's match?"
    TACTICAL = "tactical"               # "What formation does Barcelona use?"
    HISTORICAL = "historical"           # "How did Arsenal perform last season?"
    LIVE_DATA = "live_data"            # "What's the current score?"
    STANDINGS = "standings"             # "Show me the Premier League table"
    TRANSFER = "transfer"               # "Transfer rumors about Haaland"
    GENERAL = "general"                 # General football knowledge questions


class Confidence(Enum):
    """Confidence levels for LLM outputs."""
    HIGH = "high"       # > 0.9
    MEDIUM = "medium"   # 0.7 - 0.9
    LOW = "low"         # 0.5 - 0.7
    UNCERTAIN = "uncertain"  # < 0.5


@dataclass
class Entity:
    """Represents an extracted entity from a query."""
    text: str
    type: str  # 'team', 'player', 'league', 'metric', 'temporal'
    start_pos: int
    end_pos: int
    confidence: float
    resolved_id: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ResolvedEntity:
    """Represents a resolved entity with canonical information."""
    original: Entity
    canonical_name: str
    entity_id: str
    entity_type: str
    api_mappings: Dict[str, str]  # Map to different API identifiers
    aliases: List[str]
    confidence: float
    disambiguation_context: Optional[str] = None

    def __post_init__(self):
        if self.api_mappings is None:
            self.api_mappings = {}
        if self.aliases is None:
            self.aliases = []


@dataclass
class TemporalScope:
    """Represents temporal context for queries."""
    period_type: str  # 'current', 'last_season', 'specific_date', 'range'
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    season: Optional[str] = None
    relative_description: Optional[str] = None


@dataclass
class QueryIntent:
    """Structured representation of a processed query."""
    query_type: QueryType
    confidence: Confidence
    entities: List[ResolvedEntity]
    temporal_scope: Optional[TemporalScope]
    complexity: str  # 'simple', 'moderate', 'complex'
    requires_live_data: bool
    requires_historical_data: bool
    requires_prediction: bool
    original_query: str
    processed_at: datetime
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.processed_at is None:
            self.processed_at = datetime.now()


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    query: str
    intent: QueryIntent
    entities: List[ResolvedEntity]
    response: str
    timestamp: datetime
    football_context: Dict[str, Any]
    confidence: float

    def __post_init__(self):
        if self.football_context is None:
            self.football_context = {}


@dataclass
class ConversationContext:
    """Maintains conversation state and context."""
    conversation_id: str
    turns: List[ConversationTurn]
    current_focus: Dict[str, Any]  # Current teams, leagues, players in focus
    user_preferences: Dict[str, Any]
    session_metadata: Dict[str, Any]
    created_at: datetime
    last_updated: datetime

    def __post_init__(self):
        if self.turns is None:
            self.turns = []
        if self.current_focus is None:
            self.current_focus = {}
        if self.user_preferences is None:
            self.user_preferences = {}
        if self.session_metadata is None:
            self.session_metadata = {}


# ============================================================================
# Query Processor - Natural Language Understanding
# ============================================================================

class QueryProcessor:
    """
    Handles natural language understanding and intent classification.
    Transforms natural language queries into structured QueryIntent objects.
    """
    
    def __init__(self):
        self.primary_model = "gpt-4-turbo"
        self.fallback_model = "gpt-3.5-turbo"
        self.cache = {}  # Simple in-memory cache
        logger.info("QueryProcessor initialized")

    async def process(self, query: str, context: Optional[ConversationContext] = None) -> QueryIntent:
        """
        Transform natural language into structured query intent.
        
        Args:
            query: The natural language query
            context: Optional conversation context for better understanding
            
        Returns:
            QueryIntent object with classification and extracted entities
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, context)
            if cache_key in self.cache:
                logger.info(f"Cache hit for query: {query[:50]}...")
                return self.cache[cache_key]

            # Step 1: Classify intent
            intent_classification = await self._classify_intent(query, context)
            
            # Step 2: Extract and resolve entities
            entities = await self._extract_and_resolve_entities(query, intent_classification, context)
            
            # Step 3: Determine temporal scope
            temporal_scope = await self._extract_temporal_scope(query, context)
            
            # Step 4: Assess complexity and requirements
            complexity = self._assess_complexity(query, entities, intent_classification)
            requirements = self._determine_requirements(intent_classification, entities)
            
            # Create QueryIntent object
            query_intent = QueryIntent(
                query_type=intent_classification['query_type'],
                confidence=intent_classification['confidence'],
                entities=entities,
                temporal_scope=temporal_scope,
                complexity=complexity,
                requires_live_data=requirements['live_data'],
                requires_historical_data=requirements['historical_data'],
                requires_prediction=requirements['prediction'],
                original_query=query,
                processed_at=datetime.now()
            )
            
            # Cache the result
            self.cache[cache_key] = query_intent
            
            logger.info(f"Processed query: {query_intent.query_type.value}, confidence: {query_intent.confidence.value}")
            return query_intent
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {str(e)}")
            # Return a fallback intent
            return self._create_fallback_intent(query)

    async def _classify_intent(self, query: str, context: Optional[ConversationContext] = None) -> Dict[str, Any]:
        """Classify the intent of the query using LLM."""
        
        # Build context-aware prompt
        context_info = ""
        if context and context.current_focus:
            context_info = f"Current conversation focus: {context.current_focus}"
        
        classification_prompt = f"""
        You are a football query classifier. Analyze the user's question and classify it accurately.
        
        Query Types:
        - STATISTICAL: Requests for specific stats, numbers, metrics (goals, assists, points, etc.)
        - COMPARATIVE: Comparing teams, players, or performance
        - PREDICTIVE: Asking for predictions, forecasts, or future outcomes
        - TACTICAL: Questions about formations, playing style, tactics, strategy
        - HISTORICAL: Questions about past events, seasons, records
        - LIVE_DATA: Real-time scores, ongoing matches, current status
        - STANDINGS: League tables, positions, rankings
        - TRANSFER: Transfer news, rumors, market activity
        - GENERAL: General football knowledge, rules, concepts
        
        {context_info}
        
        Query: "{query}"
        
        Return ONLY a JSON object with:
        {{
            "query_type": "STATISTICAL",
            "confidence": "HIGH|MEDIUM|LOW|UNCERTAIN", 
            "reasoning": "brief explanation",
            "complexity": "simple|moderate|complex",
            "requires_live_data": false,
            "requires_historical_data": false,
            "requires_prediction": false
        }}
        """
        
        try:
            # Use existing query_llm function but extract JSON
            response = query_llm(
                question=classification_prompt,
                scores={},
                standings={},
                history=[],
                focus=""
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    'query_type': QueryType(result.get('query_type', 'general').lower()),
                    'confidence': Confidence(result.get('confidence', 'medium').lower()),
                    'reasoning': result.get('reasoning', ''),
                    'complexity': result.get('complexity', 'moderate'),
                    'requires_live_data': result.get('requires_live_data', False),
                    'requires_historical_data': result.get('requires_historical_data', False),
                    'requires_prediction': result.get('requires_prediction', False)
                }
            else:
                # Fallback to rule-based classification
                return self._rule_based_classification(query)
                
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, using rule-based fallback")
            return self._rule_based_classification(query)

    def _rule_based_classification(self, query: str) -> Dict[str, Any]:
        """Fallback rule-based classification when LLM fails."""
        query_lower = query.lower()
        
        # Simple keyword-based classification
        if any(word in query_lower for word in ['score', 'live', 'current', 'now', 'today']):
            query_type = QueryType.LIVE_DATA
        # Prefer statistical detection before standings to avoid printing full tables for team stats
        elif any(word in query_lower for word in ['goals', 'assists', 'points', 'stats', 'gd', 'goal difference', 'ga', 'gf']):
            query_type = QueryType.STATISTICAL
        elif any(word in query_lower for word in ['table', 'standings', 'position', 'rank']):
            query_type = QueryType.STANDINGS
        elif any(word in query_lower for word in ['formation', 'tactic', 'style', 'strategy']):
            query_type = QueryType.TACTICAL
        elif any(word in query_lower for word in ['predict', 'forecast', 'will', 'future']):
            query_type = QueryType.PREDICTIVE
        elif any(word in query_lower for word in ['transfer', 'buy', 'sell', 'sign']):
            query_type = QueryType.TRANSFER
        elif any(word in query_lower for word in ['vs', 'versus', 'better', 'compare']):
            query_type = QueryType.COMPARATIVE
        elif any(word in query_lower for word in ['last season', 'history', 'past', 'previous']):
            query_type = QueryType.HISTORICAL
        elif any(word in query_lower for word in ['goals', 'assists', 'points', 'stats']):
            query_type = QueryType.STATISTICAL
        else:
            query_type = QueryType.GENERAL
            
        return {
            'query_type': query_type,
            'confidence': Confidence.MEDIUM,
            'reasoning': 'Rule-based classification',
            'complexity': 'moderate',
            'requires_live_data': query_type == QueryType.LIVE_DATA,
            'requires_historical_data': query_type == QueryType.HISTORICAL,
            'requires_prediction': query_type == QueryType.PREDICTIVE
        }

    async def _extract_and_resolve_entities(self, query: str, intent: Dict[str, Any], 
                                           context: Optional[ConversationContext] = None) -> List[ResolvedEntity]:
        """Extract and resolve entities from the query."""
        entities = []
        
        try:
            # Use LLM for entity extraction
            extraction_prompt = f"""
            Extract football entities from this query. Focus on teams, players, leagues, and metrics.
            
            Query: "{query}"
            Intent: {intent.get('query_type', '').value if hasattr(intent.get('query_type', ''), 'value') else intent.get('query_type', '')}
            
            Return JSON array of entities:
            [
                {{
                    "text": "extracted text",
                    "type": "team|player|league|metric|temporal",
                    "confidence": 0.95
                }}
            ]
            """
            
            response = query_llm(
                question=extraction_prompt,
                scores={},
                standings={},
                history=[],
                focus=""
            )
            
            # Extract JSON array from response
            json_match = re.search(r'\[[^\]]+\]', response)
            if json_match:
                raw_entities = json.loads(json_match.group())
                
                for i, entity_data in enumerate(raw_entities):
                    entity = Entity(
                        text=entity_data.get('text', ''),
                        type=entity_data.get('type', 'unknown'),
                        start_pos=0,  # Simplified for now
                        end_pos=len(entity_data.get('text', '')),
                        confidence=entity_data.get('confidence', 0.8)
                    )
                    
                    # Resolve entity to canonical form
                    resolved = await self._resolve_entity(entity, query, context)
                    if resolved:
                        entities.append(resolved)
            
        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}, using regex fallback")
            entities = self._regex_entity_extraction(query)
        
        return entities

    def _regex_entity_extraction(self, query: str) -> List[ResolvedEntity]:
        """Fallback regex-based entity extraction."""
        entities = []
        
        # Simple team name patterns (this could be expanded)
        team_patterns = [
            r'\b(?:manchester\s+united|man\s+united|united)\b',
            r'\b(?:real\s+madrid|madrid)\b',
            r'\b(?:fc\s+barcelona|barcelona|barca)\b',
            r'\b(?:liverpool|arsenal|chelsea|tottenham)\b',
            r'\b(?:bayern\s+munich|bayern)\b',
        ]
        
        for pattern in team_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                entity = Entity(
                    text=match.group(),
                    type='team',
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.7
                )
                
                # Create a simple resolved entity
                resolved = ResolvedEntity(
                    original=entity,
                    canonical_name=entity.text.title(),
                    entity_id=entity.text.lower().replace(' ', '_'),
                    entity_type='team',
                    api_mappings={},
                    aliases=[entity.text],
                    confidence=0.7
                )
                entities.append(resolved)
        
        return entities

    async def _resolve_entity(self, entity: Entity, query: str, 
                             context: Optional[ConversationContext] = None) -> Optional[ResolvedEntity]:
        """Resolve entity to canonical form with API mappings."""
        
        if entity.type == 'team':
            return await self._resolve_team_entity(entity, query, context)
        elif entity.type == 'league':
            return await self._resolve_league_entity(entity, query, context)
        elif entity.type == 'player':
            return await self._resolve_player_entity(entity, query, context)
        elif entity.type == 'metric':
            return await self._resolve_metric_entity(entity, query, context)
        else:
            # Generic resolution
            return ResolvedEntity(
                original=entity,
                canonical_name=entity.text,
                entity_id=entity.text.lower().replace(' ', '_'),
                entity_type=entity.type,
                api_mappings={},
                aliases=[entity.text],
                confidence=entity.confidence
            )

    async def _resolve_team_entity(self, entity: Entity, query: str, 
                                  context: Optional[ConversationContext] = None) -> Optional[ResolvedEntity]:
        """Resolve team entity using existing team resolution logic."""
        try:
            # Use existing team resolution from main.py
            from main import resolve_team_and_fd_code
            
            result = resolve_team_and_fd_code(entity.text)
            if result:
                team_name, fd_code = result
                return ResolvedEntity(
                    original=entity,
                    canonical_name=team_name,
                    entity_id=team_name.lower().replace(' ', '_'),
                    entity_type='team',
                    api_mappings={'football_data': fd_code},
                    aliases=[entity.text, team_name],
                    confidence=0.9
                )
        except Exception as e:
            logger.warning(f"Team resolution failed: {e}")
        
        return None

    async def _resolve_league_entity(self, entity: Entity, query: str, 
                                    context: Optional[ConversationContext] = None) -> Optional[ResolvedEntity]:
        """Resolve league entity using existing league resolution logic."""
        try:
            code = resolve_competition(entity.text)
            if code:
                return ResolvedEntity(
                    original=entity,
                    canonical_name=entity.text.title(),
                    entity_id=code,
                    entity_type='league',
                    api_mappings={'football_data': code},
                    aliases=[entity.text],
                    confidence=0.9
                )
        except Exception as e:
            logger.warning(f"League resolution failed: {e}")
        
        return None

    async def _resolve_player_entity(self, entity: Entity, query: str, 
                                    context: Optional[ConversationContext] = None) -> Optional[ResolvedEntity]:
        """Resolve player entity (simplified implementation)."""
        return ResolvedEntity(
            original=entity,
            canonical_name=entity.text.title(),
            entity_id=entity.text.lower().replace(' ', '_'),
            entity_type='player',
            api_mappings={},
            aliases=[entity.text],
            confidence=0.7
        )

    async def _resolve_metric_entity(self, entity: Entity, query: str, 
                                    context: Optional[ConversationContext] = None) -> Optional[ResolvedEntity]:
        """Resolve metric entity using existing metric logic."""
        # Use existing metric detection from main.py
        metric_aliases = {
            "goalDifference": ["gd", "goal difference", "goal diff"],
            "points": ["points", "pts"],
            "position": ["position", "rank", "place"],
            "won": ["wins", "win", "won"],
            "draw": ["draws", "draw", "ties"],
            "lost": ["losses", "lost", "defeats"],
            "playedGames": ["played", "games", "matches"],
            "goalsFor": ["goals for", "gf", "scored"],
            "goalsAgainst": ["goals against", "ga", "conceded"]
        }
        
        entity_lower = entity.text.lower()
        for canonical, aliases in metric_aliases.items():
            if entity_lower in aliases:
                return ResolvedEntity(
                    original=entity,
                    canonical_name=canonical,
                    entity_id=canonical,
                    entity_type='metric',
                    api_mappings={'football_data': canonical},
                    aliases=aliases,
                    confidence=0.9
                )
        
        return None

    async def _extract_temporal_scope(self, query: str, 
                                     context: Optional[ConversationContext] = None) -> Optional[TemporalScope]:
        """Extract temporal information from the query."""
        query_lower = query.lower()
        
        # Simple temporal extraction
        if 'last season' in query_lower:
            return TemporalScope(
                period_type='last_season',
                relative_description='last season'
            )
        elif 'this season' in query_lower or 'current season' in query_lower:
            return TemporalScope(
                period_type='current',
                relative_description='current season'
            )
        elif any(word in query_lower for word in ['today', 'now', 'current', 'live']):
            return TemporalScope(
                period_type='current',
                relative_description='current/live'
            )
        
        return None

    def _assess_complexity(self, query: str, entities: List[ResolvedEntity], 
                          intent: Dict[str, Any]) -> str:
        """Assess the complexity of the query."""
        complexity_score = 0
        
        # Entity count contributes to complexity
        complexity_score += len(entities)
        
        # Query type influences complexity
        if intent.get('query_type') in [QueryType.PREDICTIVE, QueryType.TACTICAL, QueryType.COMPARATIVE]:
            complexity_score += 2
        
        # Query length
        word_count = len(query.split())
        if word_count > 10:
            complexity_score += 1
        
        # Determine complexity level
        if complexity_score <= 2:
            return 'simple'
        elif complexity_score <= 4:
            return 'moderate'
        else:
            return 'complex'

    def _determine_requirements(self, intent: Dict[str, Any], entities: List[ResolvedEntity]) -> Dict[str, bool]:
        """Determine what data sources are required for the query."""
        return {
            'live_data': intent.get('requires_live_data', False) or intent.get('query_type') == QueryType.LIVE_DATA,
            'historical_data': intent.get('requires_historical_data', False) or intent.get('query_type') == QueryType.HISTORICAL,
            'prediction': intent.get('requires_prediction', False) or intent.get('query_type') == QueryType.PREDICTIVE
        }

    def _generate_cache_key(self, query: str, context: Optional[ConversationContext] = None) -> str:
        """Generate a cache key for the query."""
        context_str = ""
        if context and context.current_focus:
            context_str = str(sorted(context.current_focus.items()))
        
        cache_input = f"{query}|{context_str}"
        return hashlib.md5(cache_input.encode()).hexdigest()

    def _create_fallback_intent(self, query: str) -> QueryIntent:
        """Create a fallback intent when processing fails."""
        return QueryIntent(
            query_type=QueryType.GENERAL,
            confidence=Confidence.LOW,
            entities=[],
            temporal_scope=None,
            complexity='moderate',
            requires_live_data=False,
            requires_historical_data=False,
            requires_prediction=False,
            original_query=query,
            processed_at=datetime.now()
        )


# ============================================================================
# Context Manager - Conversation Memory and Context Tracking
# ============================================================================

class ContextManager:
    """
    Manages conversation context, memory, and football domain knowledge.
    Provides intelligent reference resolution and context tracking.
    """
    
    def __init__(self):
        self.conversations: Dict[str, ConversationContext] = {}
        self.max_turns = 20
        self.context_expiry_hours = 24
        logger.info("ContextManager initialized")

    async def get_context(self, conversation_id: str) -> ConversationContext:
        """Retrieve or create conversation context."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationContext(
                conversation_id=conversation_id,
                turns=[],
                current_focus={},
                user_preferences={},
                session_metadata={},
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
        
        context = self.conversations[conversation_id]
        
        # Clean up old contexts
        await self._cleanup_expired_contexts()
        
        return context

    async def update_context(self, conversation_id: str, query: str, intent: QueryIntent,
                           entities: List[ResolvedEntity], response: str) -> ConversationContext:
        """Update conversation context with new turn."""
        context = await self.get_context(conversation_id)
        
        # Extract football context from the turn
        football_context = await self._extract_football_context(query, entities, response)
        
        # Create new conversation turn
        turn = ConversationTurn(
            query=query,
            intent=intent,
            entities=entities,
            response=response,
            timestamp=datetime.now(),
            football_context=football_context,
            confidence=intent.confidence.value if hasattr(intent.confidence, 'value') else 0.8
        )
        
        # Add turn to context
        context.turns.append(turn)
        
        # Update current focus based on the turn
        await self._update_current_focus(context, turn)
        
        # Manage conversation length
        if len(context.turns) > self.max_turns:
            # Keep recent turns and compress older ones
            context.turns = context.turns[-self.max_turns:]
        
        # Update timestamps
        context.last_updated = datetime.now()
        
        logger.info(f"Updated context for conversation {conversation_id}, turns: {len(context.turns)}")
        return context

    async def resolve_references(self, query: str, context: ConversationContext) -> Dict[str, Any]:
        """Resolve implicit references using conversation context."""
        references = {}
        query_lower = query.lower()
        
        # Resolve pronoun references
        if any(pronoun in query_lower for pronoun in ['they', 'them', 'their']):
            if 'teams' in context.current_focus and context.current_focus['teams']:
                references['team_reference'] = context.current_focus['teams'][-1]
        
        if any(pronoun in query_lower for pronoun in ['he', 'him', 'his']):
            if 'players' in context.current_focus and context.current_focus['players']:
                references['player_reference'] = context.current_focus['players'][-1]
        
        # Resolve temporal references
        if 'last season' in query_lower:
            current_year = datetime.now().year
            references['season'] = f"{current_year-1}/{current_year}"
        
        if 'this season' in query_lower:
            current_year = datetime.now().year
            references['season'] = f"{current_year}/{current_year+1}"
        
        # Resolve league context
        if 'league' in context.current_focus:
            references['league'] = context.current_focus['league']
        
        # Resolve "the derby" or "el clasico" type references
        if any(term in query_lower for term in ['derby', 'clasico', 'clásico']):
            if 'teams' in context.current_focus:
                references['rivalry_match'] = True
        
        return references

    async def _extract_football_context(self, query: str, entities: List[ResolvedEntity], 
                                       response: str) -> Dict[str, Any]:
        """Extract football-specific context from a conversation turn."""
        football_context = {
            'teams': [],
            'players': [],
            'leagues': [],
            'topics': [],
            'metrics': []
        }
        
        # Extract from entities
        for entity in entities:
            if entity.entity_type == 'team':
                football_context['teams'].append(entity.canonical_name)
            elif entity.entity_type == 'player':
                football_context['players'].append(entity.canonical_name)
            elif entity.entity_type == 'league':
                football_context['leagues'].append(entity.canonical_name)
            elif entity.entity_type == 'metric':
                football_context['metrics'].append(entity.canonical_name)
        
        # Extract topics from query
        query_lower = query.lower()
        if any(word in query_lower for word in ['formation', 'tactic', 'strategy']):
            football_context['topics'].append('tactics')
        if any(word in query_lower for word in ['transfer', 'sign', 'buy']):
            football_context['topics'].append('transfers')
        if any(word in query_lower for word in ['injury', 'injured', 'fitness']):
            football_context['topics'].append('injuries')
        
        # Heuristic: parse team from assistant stat-style responses like
        # "Team Name goals conceded: 8"
        try:
            if response:
                import re
                m = re.match(r"^(.*?)\s+(GD|points|position|wins|draws|losses|played|goals scored|goals conceded|form)\s*:\s*", response, flags=re.I)
                if m:
                    team = m.group(1).strip()
                    if team:
                        football_context['teams'].append(team)
        except Exception:
            pass

        return football_context

    async def _update_current_focus(self, context: ConversationContext, turn: ConversationTurn):
        """Update the current focus based on the latest turn."""
        fc = turn.football_context
        
        # Update team focus (keep last 3 mentioned teams)
        if fc.get('teams'):
            current_teams = context.current_focus.get('teams', [])
            for team in fc['teams']:
                if team not in current_teams:
                    current_teams.append(team)
            context.current_focus['teams'] = current_teams[-3:]
        
        # Update league focus (keep the most recent)
        if fc.get('leagues'):
            context.current_focus['league'] = fc['leagues'][-1]
        
        # Update player focus (keep last 3 mentioned players)
        if fc.get('players'):
            current_players = context.current_focus.get('players', [])
            for player in fc['players']:
                if player not in current_players:
                    current_players.append(player)
            context.current_focus['players'] = current_players[-3:]
        
        # Update topic focus
        if fc.get('topics'):
            context.current_focus['current_topics'] = fc['topics']
        
        # Add temporal context if present
        if turn.intent.temporal_scope:
            context.current_focus['temporal_context'] = {
                'period_type': turn.intent.temporal_scope.period_type,
                'description': turn.intent.temporal_scope.relative_description
            }

    async def _cleanup_expired_contexts(self):
        """Remove expired conversation contexts."""
        current_time = datetime.now()
        expired_conversations = []
        
        for conv_id, context in self.conversations.items():
            if current_time - context.last_updated > timedelta(hours=self.context_expiry_hours):
                expired_conversations.append(conv_id)
        
        for conv_id in expired_conversations:
            del self.conversations[conv_id]
            logger.info(f"Cleaned up expired conversation: {conv_id}")

    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get a summary of the conversation for debugging/monitoring."""
        if conversation_id not in self.conversations:
            return {"error": "Conversation not found"}
        
        context = self.conversations[conversation_id]
        return {
            "conversation_id": conversation_id,
            "created_at": context.created_at.isoformat(),
            "last_updated": context.last_updated.isoformat(),
            "turns_count": len(context.turns),
            "current_focus": context.current_focus,
            "recent_queries": [turn.query for turn in context.turns[-3:]]
        }


# ============================================================================
# Intelligence Coordinator - Main Orchestrator
# ============================================================================

class IntelligenceCoordinator:
    """
    Main orchestrator that coordinates all LLM-driven operations.
    Replaces the dictionary-based processing with intelligent routing and response synthesis.
    """
    
    def __init__(self):
        self.query_processor = QueryProcessor()
        self.context_manager = ContextManager()
        self.response_cache = {}
        self.cache_ttl_minutes = 30
        logger.info("IntelligenceCoordinator initialized")

    async def process_intelligent_query(self, query: str, conversation_id: str, 
                                       user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main entry point for intelligent query processing.
        
        Args:
            query: The user's natural language query
            conversation_id: Unique conversation identifier
            user_preferences: Optional user preferences for response customization
            
        Returns:
            Dict containing the response and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Get conversation context
            context = await self.context_manager.get_context(conversation_id)
            
            # Step 2: Process query to understand intent
            intent = await self.query_processor.process(query, context)
            
            # Step 3: Resolve references using context
            references = await self.context_manager.resolve_references(query, context)
            
            # Step 4: Check cache for similar queries
            cache_key = self._generate_response_cache_key(intent, references)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                logger.info(f"Cache hit for processed query: {query[:50]}...")
                return self._format_response(cached_response, intent, True, time.time() - start_time)
            
            # Step 5: Route to appropriate processing based on intent
            response_data = await self._route_and_process(intent, references, context)
            
            # Step 6: Generate intelligent response
            response = await self._generate_response(intent, response_data, context, references)
            
            # Step 7: Update context with this turn
            await self.context_manager.update_context(
                conversation_id, query, intent, intent.entities, response
            )
            
            # Step 8: Cache the response (store final text for quick reuse)
            self._cache_response(cache_key, response, response_data)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Successfully processed query in {processing_time:.2f}s: {intent.query_type.value}")
            
            return self._format_response(response, intent, False, processing_time)
            
        except Exception as e:
            logger.error(f"Error in intelligent query processing: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error processing your query. Please try again.",
                "error": str(e),
                "conversation_id": conversation_id,
                "processing_time": time.time() - start_time,
                "cached": False
            }

    async def _route_and_process(self, intent: QueryIntent, references: Dict[str, Any], 
                                context: ConversationContext) -> Dict[str, Any]:
        """Route query to appropriate processing based on intent."""
        from llm.openrouter_llm import _detect_metric, _detect_table_request
        q = intent.original_query
        # Metric-first guard: if a stat is requested and no table keyword, treat as STATISTICAL
        try:
            if _detect_metric(q) and not _detect_table_request(q):
                return await self._handle_statistical_query(intent, references, context)
        except Exception:
            pass

        if intent.query_type == QueryType.LIVE_DATA:
            return await self._handle_live_data_query(intent, references)
        
        elif intent.query_type == QueryType.STANDINGS:
            # Only allow standings flow if the question actually asks for a table/standings
            try:
                if not _detect_table_request(q):
                    # Reroute to general handling when no explicit table request
                    return await self._handle_general_query(intent, references, context)
            except Exception:
                pass
            return await self._handle_standings_query(intent, references)
        
        elif intent.query_type == QueryType.STATISTICAL:
            return await self._handle_statistical_query(intent, references, context)
        
        elif intent.query_type == QueryType.COMPARATIVE:
            return await self._handle_comparative_query(intent, references, context)
        
        elif intent.query_type == QueryType.TACTICAL:
            return await self._handle_tactical_query(intent, references, context)
        
        elif intent.query_type == QueryType.PREDICTIVE:
            return await self._handle_predictive_query(intent, references, context)
        
        elif intent.query_type == QueryType.HISTORICAL:
            return await self._handle_historical_query(intent, references, context)
        
        elif intent.query_type == QueryType.TRANSFER:
            return await self._handle_transfer_query(intent, references, context)
        
        else:
            return await self._handle_general_query(intent, references, context)

    async def _handle_live_data_query(self, intent: QueryIntent, references: Dict[str, Any]) -> Dict[str, Any]:
        """Handle live data queries."""
        try:
            live_scores = get_live_scores_lsa()
            return {
                "type": "live_data",
                "data": live_scores,
                "source": "livescore_api",
                "confidence": 0.95
            }
        except Exception as e:
            logger.error(f"Error fetching live scores: {e}")
            return {
                "type": "live_data",
                "data": {},
                "error": str(e),
                "confidence": 0.0
            }

    async def _handle_standings_query(self, intent: QueryIntent, references: Dict[str, Any]) -> Dict[str, Any]:
        """Handle standings/table queries."""
        try:
            # Determine league from entities or references
            league_code = None
            
            # Check entities for league
            for entity in intent.entities:
                if entity.entity_type == 'league' and entity.api_mappings.get('football_data'):
                    league_code = entity.api_mappings['football_data']
                    break
            
            # Check references for league context
            if not league_code:
                if references.get('league'):
                    league_code = references['league']
            if not league_code:
                # Try to parse from the original query text directly
                from llm.openrouter_llm import guess_league_code_from_text
                league_code = guess_league_code_from_text(intent.original_query)
            if not league_code:
                return {
                    "type": "standings",
                    "data": {},
                    "error": "League not specified",
                    "confidence": 0.0
                }
            
            standings = get_standings(league_code)
            return {
                "type": "standings",
                "data": standings,
                "league_code": league_code,
                "source": "football_data",
                "confidence": 0.9
            }
        except Exception as e:
            logger.error(f"Error fetching standings: {e}")
            return {
                "type": "standings",
                "data": {},
                "error": str(e),
                "confidence": 0.0
            }

    async def _handle_statistical_query(self, intent: QueryIntent, references: Dict[str, Any], 
                                       context: ConversationContext) -> Dict[str, Any]:
        """Handle statistical queries using existing logic."""
        try:
            # Use existing statistical processing from openrouter_llm.py
            # This integrates with the current stat detection and team resolution
            from llm.openrouter_llm import (
                try_answer_stat_from_football_data,
                _detect_metric,
                _league_name_for_code,
                guess_league_code_from_text,
            )

            result = try_answer_stat_from_football_data(intent.original_query)
            if result:
                return {
                    "type": "statistical",
                    "data": {"answer": result},
                    "source": "football_data_stats",
                    "confidence": 0.9
                }
            else:
                # Follow-up friendly fallback: synthesize missing league/metric from context
                q = intent.original_query.strip().lower()
                # Guess metric from this query or prior user turns
                metric = _detect_metric(q)
                if not metric:
                    for turn in reversed(context.turns):
                        m = _detect_metric(turn.query.lower())
                        if m:
                            metric = m
                            break
                if not metric:
                    metric = ("goalsFor", "goals scored")
                canon, label = metric

                # Guess league from references, this query, or prior turns
                league_code = references.get('league') if isinstance(references.get('league', None), str) else None
                if not league_code:
                    league_code = guess_league_code_from_text(q)
                if not league_code:
                    for turn in reversed(context.turns):
                        lc = guess_league_code_from_text(turn.query)
                        if lc:
                            league_code = lc
                            break
                # Do not default league; require it from question or context

                # Build a synthetic question using the same wording style
                # Strip trivial words like "and", "what about"
                team_guess = re.sub(r"[?!.]", " ", q)
                team_guess = re.sub(r"\b(and|what about|about|pls|please|team|they|them|their)\b", " ", team_guess)
                team_guess = re.sub(r"\s+", " ", team_guess).strip()
                if not team_guess or len(team_guess.split()) <= 2:
                    # Use last focused team from context if available
                    last_teams = context.current_focus.get('teams') or []
                    if last_teams:
                        team_guess = last_teams[-1]
                if team_guess:
                    synthetic = f"{team_guess} {label} in {_league_name_for_code(league_code) or league_code}"
                    result2 = try_answer_stat_from_football_data(synthetic)
                    if result2:
                        return {
                            "type": "statistical",
                            "data": {"answer": result2},
                            "source": "football_data_stats_followup",
                            "confidence": 0.85
                        }

                # Fallback to general LLM processing
                return await self._handle_general_query(intent, references, context)
                
        except Exception as e:
            logger.error(f"Error handling statistical query: {e}")
            return await self._handle_general_query(intent, references, context)

    async def _handle_comparative_query(self, intent: QueryIntent, references: Dict[str, Any], 
                                       context: ConversationContext) -> Dict[str, Any]:
        """Handle comparative queries."""
        # For now, delegate to general LLM processing
        return await self._handle_general_query(intent, references, context)

    async def _handle_tactical_query(self, intent: QueryIntent, references: Dict[str, Any], 
                                    context: ConversationContext) -> Dict[str, Any]:
        """Handle tactical analysis queries."""
        # For now, delegate to general LLM processing with tactical context
        return await self._handle_general_query(intent, references, context)

    async def _handle_predictive_query(self, intent: QueryIntent, references: Dict[str, Any], 
                                      context: ConversationContext) -> Dict[str, Any]:
        """Handle predictive analysis queries."""
        # Build a focused standings context for the league mentioned in this question
        try:
            from llm.openrouter_llm import guess_league_code_from_text
            code = guess_league_code_from_text(intent.original_query)
            if not code and isinstance(references.get('league', None), str):
                code = references['league']
            if not code:
                # fallback to last focused league if available
                code = context.current_focus.get('league')
            data = {}
            if code:
                try:
                    data = {code: get_standings(code)}
                except Exception:
                    data = {}
            return {
                "type": "llm_general",
                "data": {
                    "scores": {},
                    "standings": data or {},
                },
                "source": "predictive_llm",
                "confidence": 0.7,
            }
        except Exception:
            return await self._handle_general_query(intent, references, context)

    async def _handle_historical_query(self, intent: QueryIntent, references: Dict[str, Any], 
                                      context: ConversationContext) -> Dict[str, Any]:
        """Handle historical analysis queries."""
        # For now, delegate to general LLM processing
        return await self._handle_general_query(intent, references, context)

    async def _handle_transfer_query(self, intent: QueryIntent, references: Dict[str, Any], 
                                    context: ConversationContext) -> Dict[str, Any]:
        """Handle transfer-related queries."""
        # For now, delegate to general LLM processing
        return await self._handle_general_query(intent, references, context)

    async def _handle_general_query(self, intent: QueryIntent, references: Dict[str, Any], 
                                   context: ConversationContext) -> Dict[str, Any]:
        """Handle general queries using LLM."""
        try:
            # Prepare data for LLM
            scores_data = {}
            standings_data = {}
            
            # Get live scores if needed
            if intent.requires_live_data:
                try:
                    scores_data = get_live_scores_lsa()
                except Exception as e:
                    logger.warning(f"Failed to fetch live scores: {e}")
            
            # Get standings data
            league_codes = ["PL"]  # Default
            
            # Extract league from context or entities
            if references.get('league'):
                league_codes = [references['league']]
            elif context.current_focus.get('league'):
                league_codes = [context.current_focus['league']]
            
            # Add leagues from entities
            for entity in intent.entities:
                if entity.entity_type == 'league' and entity.api_mappings.get('football_data'):
                    code = entity.api_mappings['football_data']
                    if code not in league_codes:
                        league_codes.append(code)
            
            # Fetch standings for relevant leagues
            for code in league_codes[:3]:  # Limit to 3 leagues to avoid overload
                try:
                    standings_data[code] = get_standings(code)
                except Exception as e:
                    logger.warning(f"Failed to fetch standings for {code}: {e}")
            
            return {
                "type": "llm_general",
                "data": {
                    "scores": scores_data,
                    "standings": standings_data,
                    "intent": asdict(intent),
                    "references": references
                },
                "source": "llm_processing",
                "confidence": 0.8
            }
            
        except Exception as e:
            logger.error(f"Error in general query handling: {e}")
            return {
                "type": "error",
                "data": {"error": str(e)},
                "confidence": 0.0
            }

    async def _generate_response(self, intent: QueryIntent, response_data: Dict[str, Any], 
                                context: ConversationContext, references: Dict[str, Any]) -> str:
        """Generate intelligent response based on processed data."""
        
        if response_data.get("type") == "error":
            return f"I apologize, but I encountered an error: {response_data.get('data', {}).get('error', 'Unknown error')}"
        
        if response_data.get("type") == "statistical" and response_data.get("data", {}).get("answer"):
            return response_data["data"]["answer"]
        
        if response_data.get("type") == "llm_general":
            # Use existing LLM processing
            data = response_data["data"]
            
            # Build conversation history from context
            history = []
            for turn in context.turns[-5:]:  # Last 5 turns
                history.append({"role": "user", "content": turn.query})
                history.append({"role": "assistant", "content": turn.response})
            
            # Determine focus league: prefer current question > references > context
            try:
                from llm.openrouter_llm import guess_league_code_from_text
                focus_league = guess_league_code_from_text(intent.original_query) or \
                               references.get('league') or \
                               context.current_focus.get('league', '')
            except Exception:
                focus_league = references.get('league') or context.current_focus.get('league', '')
            
            try:
                response = query_llm(
                    question=intent.original_query,
                    scores=data.get("scores", {}),
                    standings=data.get("standings", {}),
                    history=history,
                    focus=focus_league
                )
                return response.strip() if response else "I couldn't generate a response for that query."
            except Exception as e:
                logger.error(f"LLM response generation failed: {e}")
                return "I apologize, but I couldn't process that query at the moment."
        
        # Fallback for other types
        if response_data.get("type") == "live_data":
            return "Here are the current live scores: " + str(response_data.get("data", {}))
        
        if response_data.get("type") == "standings":
            try:
                from llm.openrouter_llm import format_league_table
                std = response_data.get("data") or {}
                code = response_data.get("league_code", "") or ""
                if std:
                    return format_league_table(std, code or "")
            except Exception:
                pass
            # Fallback minimal message
            return f"Here are the standings for {response_data.get('league_code', 'the league')} (data unavailable)"
        
        return "I processed your query but couldn't generate an appropriate response."

    def _generate_response_cache_key(self, intent: QueryIntent, references: Dict[str, Any]) -> str:
        """Generate cache key for responses."""
        key_data = {
            "query_type": intent.query_type.value,
            "entities": [{"type": e.entity_type, "id": e.entity_id} for e in intent.entities],
            "references": references,
            "complexity": intent.complexity
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Retrieve cached response if not expired."""
        if cache_key in self.response_cache:
            cached_data = self.response_cache[cache_key]
            if datetime.now() - cached_data["timestamp"] < timedelta(minutes=self.cache_ttl_minutes):
                return cached_data.get("response")
            else:
                # Remove expired cache
                del self.response_cache[cache_key]
        return None

    def _cache_response(self, cache_key: str, response_text: str, response_data: Dict[str, Any]):
        """Cache response data and final response text."""
        self.response_cache[cache_key] = {
            "response": response_text,
            "response_data": response_data,
            "timestamp": datetime.now()
        }

    def _format_response(self, response: str, intent: QueryIntent, cached: bool, processing_time: float) -> Dict[str, Any]:
        """Format the final response."""
        return {
            "answer": response,
            "conversation_id": intent.processed_at.strftime("%Y%m%d_%H%M%S"),  # Simplified for demo
            "intent": {
                "type": intent.query_type.value,
                "confidence": intent.confidence.value,
                "complexity": intent.complexity
            },
            "entities": [
                {
                    "text": e.original.text,
                    "type": e.entity_type,
                    "canonical_name": e.canonical_name
                } for e in intent.entities
            ],
            "processing_time": round(processing_time, 3),
            "cached": cached,
            "requires_live_data": intent.requires_live_data,
            "requires_historical_data": intent.requires_historical_data,
            "requires_prediction": intent.requires_prediction
        }

    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status for monitoring."""
        return {
            "query_processor": "active",
            "context_manager": "active",
            "active_conversations": len(self.context_manager.conversations),
            "cache_size": len(self.response_cache),
            "total_queries_processed": "N/A",  # Would need persistent storage
            "uptime": "N/A"  # Would need startup tracking
        }


# ============================================================================
# Factory Functions and Error Handling
# ============================================================================

class IntelligenceEngineError(Exception):
    """Base exception for intelligence engine errors."""
    pass


class QueryProcessingError(IntelligenceEngineError):
    """Error in query processing."""
    pass


class ContextManagementError(IntelligenceEngineError):
    """Error in context management."""
    pass


def create_intelligence_engine() -> IntelligenceCoordinator:
    """Factory function to create a configured intelligence engine."""
    try:
        coordinator = IntelligenceCoordinator()
        logger.info("Intelligence engine created successfully")
        return coordinator
    except Exception as e:
        logger.error(f"Failed to create intelligence engine: {e}")
        raise IntelligenceEngineError(f"Failed to initialize intelligence engine: {e}")


# ============================================================================
# Integration Helper Functions
# ============================================================================

async def process_query_with_intelligence(query: str, conversation_id: str, 
                                         coordinator: Optional[IntelligenceCoordinator] = None) -> Dict[str, Any]:
    """
    Convenience function for processing queries with the intelligence engine.
    Can be used as a drop-in replacement for existing query processing.
    """
    if coordinator is None:
        coordinator = create_intelligence_engine()
    
    return await coordinator.process_intelligent_query(query, conversation_id)


def migrate_existing_conversation(conversation_data: Dict[str, Any], 
                                 coordinator: IntelligenceCoordinator) -> str:
    """
    Migrate existing conversation data to the new intelligence engine format.
    """
    # This would be used during migration to convert existing conversation stores
    # Implementation would depend on the existing data format
    pass


# Example usage and testing
if __name__ == "__main__":
    async def test_intelligence_engine():
        """Test function for the intelligence engine."""
        coordinator = create_intelligence_engine()
        
        test_queries = [
            "What's the current Premier League table?",
            "How many goals has Messi scored this season?",
            "Who will win Barcelona vs Real Madrid?",
            "What formation does Arsenal use?",
            "Show me live scores"
        ]
        
        for query in test_queries:
            print(f"\nTesting query: {query}")
            result = await coordinator.process_intelligent_query(query, "test_conversation")
            print(f"Response: {result.get('answer', 'No answer')}")
            print(f"Intent: {result.get('intent', {}).get('type', 'Unknown')}")
            print(f"Processing time: {result.get('processing_time', 0):.3f}s")
    
    # Run test if executed directly
    import asyncio
    asyncio.run(test_intelligence_engine())
