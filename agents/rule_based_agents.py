#!/usr/bin/env python3
"""
Rule-based poker agents with different playing styles
These provide more challenging opponents than random players
"""

import random
import numpy as np
from typing import Optional, List
from agents.base_agent import BaseAgent


class TightAggressiveAgent(BaseAgent):
    """
    Tight-Aggressive (TAG) bot:
    - Plays premium hands only
    - Bets and raises aggressively when in hand
    - Good post-flop aggression
    """
    
    def __init__(self, env, name: str = "TAG_Bot"):
        super().__init__(name)
        self.env = env
        self.style = "Tight-Aggressive"
        
        # Tight hand selection - only premium hands
        self.premium_hands = {
            'AA', 'KK', 'QQ', 'JJ', 'TT', '99',
            'AKs', 'AQs', 'AJs', 'ATs', 'A9s',
            'AK', 'AQ', 'AJ', 'KQs', 'KJs'
        }
        
        # Aggression factors
        self.preflop_raise_chance = 0.8  # Raise premium hands 80% of time
        self.postflop_bet_chance = 0.7   # Bet 70% when have decent hand
        self.bluff_chance = 0.15         # Bluff 15% of time
    
    def get_hand_strength(self, hole_cards, community_cards=None):
        """Evaluate hand strength (simplified)"""
        if not hole_cards or len(hole_cards) != 2:
            return 0.0
        
        # Convert to simple notation
        card1, card2 = hole_cards
        suited = 's' if card1.suit == card2.suit else ''
        
        # Normalize ranks
        rank1 = self._rank_to_value(card1.rank)
        rank2 = self._rank_to_value(card2.rank)
        
        # Make consistent hand notation
        if rank1 >= rank2:
            hand = f"{self._value_to_rank(rank1)}{self._value_to_rank(rank2)}{suited}"
        else:
            hand = f"{self._value_to_rank(rank2)}{self._value_to_rank(rank1)}{suited}"
        
        if hand in self.premium_hands:
            return 0.9  # Premium hand
        elif rank1 >= 10 or rank2 >= 10:  # Face cards
            return 0.6
        elif rank1 == rank2:  # Pocket pair
            return 0.7
        else:
            return 0.3
    
    def _rank_to_value(self, rank):
        """Convert rank to numeric value"""
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                   '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        return rank_map.get(rank, 2)
    
    def _value_to_rank(self, value):
        """Convert numeric value back to rank"""
        value_map = {2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
                    9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
        return value_map.get(value, '2')
    
    def act(self, observation, action_mask=None, **kwargs):
        """Make decision based on tight-aggressive strategy"""
        if action_mask is None or not any(action_mask):
            return 0  # Fold if no legal actions
        
        # Simple decision based on observation
        # observation is typically the environment state
        try:
            # Use observation to make decisions instead of trying to access internal game state
            # This is a simplified approach that works with the current environment
            
            # For now, use random decisions weighted by agent personality
            # This ensures agents aren't just checking constantly
            if action_mask[2]:  # Can raise
                if random.random() < self.preflop_raise_chance:
                    return 2
            
            if action_mask[1]:  # Can call/check
                if random.random() < 0.7:  # Call/check 70% of time
                    return 1
            
            # Default to fold if no other decision made
            return 0 if action_mask[0] else 1
            
        except Exception as e:
            # Fallback to safe action
            if action_mask[1]:  # Can call/check
                return 1
            elif action_mask[0]:  # Can fold
                return 0
            else:
                return 2  # Raise as last resort
                
        except Exception:
            # Fallback to conservative play
            if action_mask[1]:  # Call/check
                return 1
            elif action_mask[0]:  # Fold
                return 0
            else:
                return 0
    
    def get_action(self, game_state, player_state):
        """Legacy method for BaseAgent compatibility"""
        # Convert to action mask format and call act()
        action_mask = [True, True, True]  # Allow all actions by default
        return self.act(None, action_mask=action_mask)


class LooseAggressiveAgent(BaseAgent):
    """
    Loose-Aggressive (LAG) bot:
    - Plays many hands
    - Very aggressive betting
    - High bluff frequency
    """
    
    def __init__(self, env, name: str = "LAG_Bot"):
        super().__init__(name)
        self.env = env
        self.style = "Loose-Aggressive"
        self.play_frequency = 0.6  # Play 60% of hands
        self.aggression = 0.8      # Very aggressive
        self.bluff_chance = 0.3    # Bluff 30% of time
    
    def act(self, observation, action_mask=None, **kwargs):
        """Make decision based on loose-aggressive strategy"""
        if action_mask is None or not any(action_mask):
            return 0
        
        try:
            # LAG plays most hands aggressively
            should_play = random.random() < self.play_frequency  # 0.7
            should_be_aggressive = random.random() < self.aggression  # 0.8
            
            if should_play:
                if should_be_aggressive and action_mask[2]:  # Raise
                    return 2
                elif action_mask[1]:  # Call
                    return 1
            
            # Sometimes bluff
            if random.random() < self.bluff_chance and action_mask[2]:
                return 2
            
            # Default decision
            if action_mask[1] and random.random() < 0.4:  # Call sometimes
                return 1
            elif action_mask[0]:  # Fold
                return 0
            else:
                return 1
                
        except Exception:
            # Fallback
            if action_mask[2] and random.random() < 0.5:
                return 2
            elif action_mask[1]:
                return 1
            else:
                return 0
    
    def get_action(self, game_state, player_state):
        """Legacy method for BaseAgent compatibility"""
        action_mask = [True, True, True]
        return self.act(None, action_mask=action_mask)


class TightPassiveAgent(BaseAgent):
    """
    Tight-Passive (Rock) bot:
    - Plays premium hands only
    - Rarely raises, prefers calling
    - Very predictable
    """
    
    def __init__(self, env, name: str = "Rock_Bot"):
        super().__init__(name)
        self.env = env
        self.style = "Tight-Passive"
        self.play_frequency = 0.2  # Play only 20% of hands
        self.raise_chance = 0.1    # Rarely raises
    
    def act(self, observation, action_mask=None, **kwargs):
        """Make decision based on tight-passive strategy"""
        if action_mask is None or not any(action_mask):
            return 0
        
        try:
            # Only play premium hands (tight)
            should_play = random.random() < self.play_frequency  # 0.25
            
            if should_play:
                # Rarely raise, prefer calling (passive)
                if action_mask[2] and random.random() < self.raise_chance:  # 0.1
                    return 2
                elif action_mask[1]:
                    return 1
            
            # Very tight - fold most hands
            if action_mask[0]:
                return 0
            elif action_mask[1]:  # Check if can't fold
                return 1
            else:
                return 0
                
        except Exception:
            # Conservative fallback
            if action_mask[1]:
                return 1
            else:
                return 0
    
    def get_action(self, game_state, player_state):
        """Legacy method for BaseAgent compatibility"""
        action_mask = [True, True, True]
        return self.act(None, action_mask=action_mask)


class LoosePassiveAgent(BaseAgent):
    """
    Loose-Passive (Fish) bot:
    - Plays many hands
    - Calls frequently
    - Rarely raises or folds
    """
    
    def __init__(self, env, name: str = "Fish_Bot"):
        super().__init__(name)
        self.env = env
        self.style = "Loose-Passive"
        self.play_frequency = 0.8  # Play 80% of hands
        self.call_frequency = 0.7  # Call 70% of time
        self.raise_chance = 0.05   # Rarely raises
    
    def act(self, observation, action_mask=None, **kwargs):
        """Make decision based on loose-passive strategy"""
        if action_mask is None or not any(action_mask):
            return 0
        
        try:
            # Fish loves to call
            should_call = random.random() < self.call_frequency
            
            if should_call and action_mask[1]:  # Call/check
                return 1
            elif action_mask[2] and random.random() < self.raise_chance:  # Rare raise
                return 2
            elif action_mask[1]:  # Default to call
                return 1
            elif action_mask[0]:  # Fold as last resort
                return 0
            else:
                return 1
                
        except Exception:
            # Default to calling
            if action_mask[1]:
                return 1
            else:
                return 0
    
    def get_action(self, game_state, player_state):
        """Legacy method for BaseAgent compatibility"""
        action_mask = [True, True, True]
        return self.act(None, action_mask=action_mask)


# Agent factory for easy creation
def create_rule_based_agents(env, count_per_type: int = 1) -> List[BaseAgent]:
    """Create a mix of rule-based agents"""
    agents = []
    
    agent_types = [
        (TightAggressiveAgent, "TAG"),
        (LooseAggressiveAgent, "LAG"), 
        (TightPassiveAgent, "Rock"),
        (LoosePassiveAgent, "Fish")
    ]
    
    for agent_class, prefix in agent_types:
        for i in range(count_per_type):
            name = f"{prefix}_{i+1}" if count_per_type > 1 else prefix
            agents.append(agent_class(env, name))
    
    return agents


def get_mixed_opponent_pool(env, total_opponents: int = 17) -> List[BaseAgent]:
    """
    Create a balanced mix of opponents for training
    
    For 17 opponents (18 total with Sharky):
    - 4 TAG bots (tight-aggressive)
    - 4 LAG bots (loose-aggressive) 
    - 4 Rock bots (tight-passive)
    - 5 Fish bots (loose-passive) - most common type
    """
    agents = []
    
    # Add TAG bots
    for i in range(4):
        agents.append(TightAggressiveAgent(env, f"TAG_{i+1}"))
    
    # Add LAG bots  
    for i in range(4):
        agents.append(LooseAggressiveAgent(env, f"LAG_{i+1}"))
    
    # Add Rock bots
    for i in range(4):
        agents.append(TightPassiveAgent(env, f"Rock_{i+1}"))
    
    # Add Fish bots (most common)
    for i in range(5):
        agents.append(LoosePassiveAgent(env, f"Fish_{i+1}"))
    
    return agents[:total_opponents]
