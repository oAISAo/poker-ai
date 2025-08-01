#!/usr/bin/env python3
"""
Simple subclass that adds rule-based opponents to existing multi-table tournament
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.multi_table_tournament_env import MultiTableTournamentEnv
from agents.rule_based_agents import get_mixed_opponent_pool
from engine.player import Player


class RuleBasedTournamentEnv(MultiTableTournamentEnv):
    """
    Simple extension of MultiTableTournamentEnv that uses rule-based opponents
    """
    
    def _setup_tournament(self):
        """Initialize all players and distribute them across tables, preserving Player_0 position"""
        # Create all players - overridden to create rule-based opponents
        self.all_players = self._create_players()
        
        # Preserve Player_0 at index 0, shuffle only the opponents
        if len(self.all_players) > 1 and self.all_players[0].name == "Player_0":
            opponents = self.all_players[1:]
            import random
            random.shuffle(opponents)
            self.all_players = [self.all_players[0]] + opponents
        else:
            # Fallback: ensure Player_0 is at index 0
            player_0 = None
            others = []
            for player in self.all_players:
                if player.name == "Player_0":
                    player_0 = player
                else:
                    others.append(player)
            import random
            random.shuffle(others)
            if player_0:
                self.all_players = [player_0] + others
        
        # Distribute players across tables
        self._distribute_players_to_tables()
        
        # Initialize tracking
        self.prev_stacks = {p.name: p.stack for p in self.all_players}
        print(f"Tournament initialized: {len(self.tables)} tables, {self.total_players} players")
    
    def step(self, action: int):
        """Execute one step, using rule-based agents for non-RL players"""
        # Get current table and player
        if self.active_table_id not in self.tables:
            return super().step(action)
            
        table = self.tables[self.active_table_id]
        if not table.players or table.game.current_player_idx >= len(table.players):
            return super().step(action)
            
        current_player = table.players[table.game.current_player_idx]
        
        # If current player is Player_0 (Sharky), use the provided action
        if current_player.name == "Player_0":
            return super().step(action)
        
        # If current player has a rule-based agent, get their decision
        if hasattr(current_player, 'agent'):
            try:
                # Get observation and action mask
                obs = self._get_obs()
                action_mask = self.legal_action_mask()
                
                # Get rule-based agent's decision
                agent_action = current_player.agent.act(obs, action_mask=action_mask)
                
                # Use the agent's action instead of the provided action
                return super().step(agent_action)
                
            except Exception as e:
                print(f"[DEBUG] Error getting rule-based agent action: {e}")
                # Fallback to safe action
                action_mask = self.legal_action_mask()
                if action_mask[1]:  # Can call/check
                    return super().step(1)
                elif action_mask[0]:  # Can fold
                    return super().step(0)
                else:
                    return super().step(2)  # Raise as last resort
        
        # Fallback to provided action
        return super().step(action)
    
    def _create_players(self):
        """Override to create mix of RL agent + rule-based opponents"""
        # Player_0 is always the RL agent (Sharky)
        players = [Player("Player_0", stack=self.starting_stack)]
        
        # Create rule-based opponents for remaining slots
        # Use self as the environment (no need for temp env)
        rule_based_agents = get_mixed_opponent_pool(self, self.total_players - 1)
        
        # Convert rule-based agents to players
        for agent in rule_based_agents:
            player = Player(agent.name, stack=self.starting_stack)
            player.agent = agent  # Store reference for decision making
            players.append(player)
        
        # Print opponent mix
        agent_types = {}
        for player in players[1:]:  # Skip Player_0 (Sharky)
            agent_type = getattr(player.agent, 'style', 'Unknown')
            agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
        print(f"Opponent mix: {dict(agent_types)}")
        
        return players


def create_rule_based_training_env(total_players=18, **kwargs):
    """Factory function to create training environment with rule-based opponents"""
    return RuleBasedTournamentEnv(
        total_players=total_players,
        max_players_per_table=9,
        min_players_per_table=2,
        table_balancing_threshold=5,
        hands_per_blind_level=9,  # Turbo tournament
        **kwargs
    )
