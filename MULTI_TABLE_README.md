# Multi-Table Poker Tournament Environment

This document describes the Multi-Table Tournament Environment, a sophisticated poker tournament simulation that supports multiple tables, automatic table balancing, realistic blind structures, and comprehensive tournament progression tracking.

## Overview

The Multi-Table Tournament Environment (`MultiTableTournamentEnv`) extends the basic poker environment to support realistic tournament play with:

- **Multiple Tables**: Automatically distributes players across multiple tables (up to 9 players per table)
- **Table Balancing**: Dynamically rebalances tables as players are eliminated
- **Blind Progression**: Realistic blind level increases on a schedule
- **Tournament Tracking**: Comprehensive elimination order and placement tracking
- **Scalable Design**: Supports tournaments from 9 to 999+ players

## Key Features

### 1. Multi-Table Support
- Automatically creates the optimal number of tables based on player count
- Supports 2-9 players per table (configurable)
- Round-robin table rotation for fair play
- Real-time table status monitoring

### 2. Table Balancing
- Monitors table sizes and triggers balancing when needed
- Moves players between tables to maintain balance
- Breaks tables when they become too small
- Preserves blind rotation and seating order

### 3. Realistic Blind Structure
- Tournament-style blind progression
- Configurable hands per blind level
- Automatic blind increases across all tables
- Standard tournament blind schedule included

### 4. Tournament Management
- Tracks player elimination order for placement rewards
- Maintains comprehensive tournament statistics
- Handles heads-up and final table scenarios
- Supports tournament completion detection

## Usage

### Basic Usage

```python
from env.multi_table_tournament_env import MultiTableTournamentEnv

# Create a 27-player tournament with 3 tables
env = MultiTableTournamentEnv(
    total_players=27,
    max_players_per_table=9,
    hands_per_blind_level=10
)

# Reset and start tournament
obs, info = env.reset()

# Play tournament
while not env._tournament_finished():
    mask = info["action_mask"]
    action = select_action(obs, mask)  # Your action selection logic
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        break

# Get final results
stats = env.get_tournament_stats()
print(f"Winner: {stats['chip_leader']}")
```

### Configuration Options

```python
env = MultiTableTournamentEnv(
    total_players=99,              # Total tournament players
    max_players_per_table=9,       # Maximum players per table
    min_players_per_table=6,       # Minimum before balancing
    starting_stack=1000,           # Starting chip stack
    hands_per_blind_level=10,      # Hands before blind increase
    table_balancing_threshold=7,   # Trigger balancing below this
    blinds_schedule=[              # Custom blind structure
        (10, 20), (15, 30), (25, 50), ...
    ]
)
```

## Architecture

### Core Classes

#### `MultiTableTournamentEnv`
Main environment class that manages the entire tournament.

**Key Methods:**
- `reset()`: Initialize/restart tournament
- `step(action)`: Execute player action
- `legal_action_mask()`: Get valid actions for current player
- `get_tournament_stats()`: Get comprehensive statistics
- `render()`: Display tournament status

#### `Table`
Represents a single poker table within the tournament.

**Key Methods:**
- `get_active_player_count()`: Count players with chips
- `add_player(player)`: Add player to table
- `remove_player(player)`: Remove player from table

### Observation Space

8-dimensional observation vector:
1. **Player Stack**: Current chip count
2. **To Call**: Amount needed to call current bet
3. **Pot**: Current pot size
4. **Current Bet**: Player's current bet amount
5. **In Hand**: Whether player is in current hand (0/1)
6. **Table ID**: Current table identifier
7. **Players at Table**: Number of players at current table
8. **Blind Level**: Current blind level (0-based)

### Action Space

3 discrete actions:
- **0**: Fold
- **1**: Call/Check
- **2**: Raise (minimum raise or all-in)

## Tournament Progression

### Phase 1: Early Tournament
- All tables active with 8-9 players each
- Regular blind increases
- Minimal table balancing needed

### Phase 2: Mid Tournament
- Some players eliminated, tables start to shrink
- Table balancing becomes active
- Some tables may be broken and players redistributed

### Phase 3: Late Tournament
- Significant eliminations
- Frequent table balancing
- Tables consolidated as field shrinks

### Phase 4: Final Table
- Down to 9 or fewer players
- Single table play
- Heads-up finale

## Training Considerations

### Challenges
- **Variable Table Sizes**: Player count per table changes during tournament
- **Table Transitions**: Players move between tables during balancing
- **Long Episodes**: Tournaments can last hundreds of hands
- **Complex State Space**: Multi-table state is more complex than single table

### Training Strategies
- **Curriculum Learning**: Start with smaller tournaments, increase size gradually
- **Multi-Agent Training**: Train multiple agents simultaneously at different tables
- **Transfer Learning**: Pre-train on single-table, fine-tune on multi-table
- **Reward Shaping**: Design rewards that account for tournament placement

### Performance Optimization
- **Parallel Tables**: Consider parallel execution of table actions
- **State Representation**: Optimize observation encoding for efficiency
- **Action Masking**: Use action masking to improve training efficiency
- **Early Stopping**: Implement tournament time limits for training

## Advanced Features

### Table Balancing Algorithm
The environment uses a sophisticated table balancing algorithm:

1. **Monitor**: Continuously track player counts at each table
2. **Trigger**: Activate balancing when tables drop below threshold
3. **Balance**: Move players from tables with excess to tables with deficit
4. **Break**: Eliminate tables that become too small
5. **Preserve**: Maintain blind rotation and fair seating

### Blind Structure
Default tournament blind structure follows standard tournament progression:

```python
blinds_schedule = [
    (10, 20),    # Level 1
    (15, 30),    # Level 2
    (25, 50),    # Level 3
    (50, 100),   # Level 4
    ...
]
```

### Statistics Tracking
Comprehensive tournament statistics:

- Player counts (remaining, eliminated)
- Table status (active tables, players per table)
- Blind progression (current level, hands played)
- Chip distribution (average stack, chip leader)
- Tournament timing (hands played, estimated duration)

## Testing

Run the test suite to verify functionality:

```bash
# Run all multi-table tests
python -m pytest test/test_multi_table_tournament_env.py -v

# Run specific test
python -m pytest test/test_multi_table_tournament_env.py::test_table_balancing_trigger -v

# Run demo
python demo_multi_table_tournament.py
```

## Training Example

Train an agent on multi-table tournaments:

```bash
# Train on 27-player tournaments
python -m train.train_multi_table_agents --agent sharky --timesteps 100000 --total-players 27

# Train on large tournaments
python -m train.train_multi_table_agents --agent sharky --timesteps 200000 --total-players 99 --tournaments 3
```

## Integration with Existing Codebase

The multi-table environment integrates seamlessly with existing components:

- **Engine**: Uses the same `PokerGame` and `Player` classes
- **Agents**: Compatible with all existing agent implementations
- **Action Validation**: Leverages existing validation logic
- **Training**: Works with existing RL training infrastructure

## Future Enhancements

Potential improvements and extensions:

1. **Satellite Tournaments**: Qualify smaller tournaments for larger events
2. **Rebuy/Add-on**: Support tournament formats with rebuys
3. **Sit-n-Go**: Single-table tournament variants
4. **Multi-Tournament**: Run multiple tournaments simultaneously
5. **Advanced Balancing**: More sophisticated table balancing algorithms
6. **Real-time Visualization**: Live tournament progress visualization

## Performance Benchmarks

Typical performance characteristics:

- **27 players**: ~200-500 hands, 1-3 hours simulated time
- **99 players**: ~800-2000 hands, 4-8 hours simulated time
- **Training Speed**: ~1000-5000 steps/second (depends on hardware)
- **Memory Usage**: ~50-200 MB per tournament instance

The Multi-Table Tournament Environment provides a realistic and comprehensive platform for training and evaluating poker AI in tournament settings, closely mimicking real-world tournament poker dynamics.
