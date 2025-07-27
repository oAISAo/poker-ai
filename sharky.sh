#!/bin/bash
# Sharky Evolution Quick Commands
# Simple wrapper for common Sharky evolution tasks


# OPTION 1:
# # Train specific versions
# python sharky_evolution_runner.py train 1.0.1
# python sharky_evolution_runner.py train 1.0.2 --from 1.0.1

# # Evaluate performance
# python sharky_evolution_runner.py evaluate 1.0.0

# # Check stats
# python sharky_evolution_runner.py stats 1.0.0

# # Train all versions at once
# python sharky_evolution_runner.py train-all 1.0


# OPTION 2:
# # Make it executable first
# chmod +x sharky.sh

# # Then use simple commands:
# ./sharky.sh train 1.0.1          # Train next version
# ./sharky.sh train-next           # Automatically train next version
# ./sharky.sh evaluate 1.0.0       # Evaluate a version
# ./sharky.sh stats 1.0.0          # Show stats
# ./sharky.sh status               # Show status of all versions
# ./sharky.sh train-all            # Train all versions 1.0.0-1.0.9
# ./sharky.sh tournament 1.0.0 1.0.1  # Run tournament between versions


# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "sharky_evolution_runner.py" ]; then
    echo -e "${RED}❌ Error: Must be run from poker-ai directory${NC}"
    exit 1
fi

# Function to display usage
show_help() {
    echo -e "${BLUE}🦈 Sharky Evolution Quick Commands${NC}"
    echo
    echo "Usage: $0 [command] [options]"
    echo
    echo "Commands:"
    echo "  train <version>              Train a specific version (e.g., 1.0.1)"
    echo "  train-next                   Train the next version automatically"
    echo "  evaluate <version>           Evaluate a version in tournaments"
    echo "  stats <version>              Show training statistics"
    echo "  train-all                    Train all versions 1.0.0 through 1.0.9"
    echo "  tournament <v1> <v2> [v3]... Run tournament between versions"
    echo "  status                       Show status of all versions"
    echo
    echo "Examples:"
    echo "  $0 train 1.0.1               # Train version 1.0.1"
    echo "  $0 train-next                # Train next version after latest"
    echo "  $0 evaluate 1.0.0            # Evaluate version 1.0.0"
    echo "  $0 stats 1.0.0               # Show stats for 1.0.0"
    echo "  $0 train-all                 # Train all versions"
    echo "  $0 tournament 1.0.0 1.0.1    # Tournament between versions"
    echo "  $0 status                    # Show what's been trained"
}

# Function to find latest trained version
find_latest_version() {
    local latest=""
    for i in {0..9}; do
        if [ -f "models/sharky_evolution/sharky_1.0.$i.zip" ]; then
            latest="1.0.$i"
        fi
    done
    echo "$latest"
}

# Function to get next version to train
get_next_version() {
    local latest=$(find_latest_version)
    if [ -z "$latest" ]; then
        echo "1.0.0"
    else
        local minor=$(echo "$latest" | cut -d'.' -f3)
        local next_minor=$((minor + 1))
        if [ $next_minor -le 9 ]; then
            echo "1.0.$next_minor"
        else
            echo ""
        fi
    fi
}

# Function to show status of all versions
show_status() {
    echo -e "${BLUE}📊 Sharky Evolution Status${NC}"
    echo
    printf "%-10s %-10s %-15s %-10s\n" "Version" "Trained" "Evaluated" "Win Rate"
    echo "----------------------------------------"
    
    for i in {0..9}; do
        version="1.0.$i"
        model_file="models/sharky_evolution/sharky_$version.zip"
        stats_file="models/sharky_evolution/sharky_${version}_stats.npy"
        
        if [ -f "$model_file" ]; then
            trained="✅"
            if [ -f "$stats_file" ]; then
                # Try to extract win rate from stats
                win_rate=$(python -c "
import numpy as np
try:
    stats = np.load('$stats_file', allow_pickle=True).item()
    if 'win_rate' in stats and stats['tournaments_played'] > 0:
        print(f\"{stats['win_rate']:.1%}\")
    else:
        print('N/A')
except:
    print('N/A')
" 2>/dev/null)
                evaluated="✅"
            else
                evaluated="❌"
                win_rate="N/A"
            fi
        else
            trained="❌"
            evaluated="❌"
            win_rate="N/A"
        fi
        
        printf "%-10s %-10s %-15s %-10s\n" "$version" "$trained" "$evaluated" "$win_rate"
    done
}

# Main command handling
case "$1" in
    "train")
        if [ -z "$2" ]; then
            echo -e "${RED}❌ Error: Please specify version to train${NC}"
            echo "Usage: $0 train <version>"
            exit 1
        fi
        echo -e "${GREEN}🦈 Training Sharky $2...${NC}"
        python sharky_evolution_runner.py train "$2"
        ;;
    
    "train-next")
        next_version=$(get_next_version)
        if [ -z "$next_version" ]; then
            echo -e "${YELLOW}🎯 All versions (1.0.0-1.0.9) already trained!${NC}"
            exit 0
        fi
        echo -e "${GREEN}🦈 Training next version: Sharky $next_version...${NC}"
        python sharky_evolution_runner.py train "$next_version"
        ;;
    
    "evaluate")
        if [ -z "$2" ]; then
            echo -e "${RED}❌ Error: Please specify version to evaluate${NC}"
            echo "Usage: $0 evaluate <version>"
            exit 1
        fi
        echo -e "${GREEN}🏆 Evaluating Sharky $2...${NC}"
        python evaluate_sharky_simple.py "$2"
        ;;
    
    "stats")
        if [ -z "$2" ]; then
            echo -e "${RED}❌ Error: Please specify version${NC}"
            echo "Usage: $0 stats <version>"
            exit 1
        fi
        echo -e "${BLUE}📊 Sharky $2 Training Stats${NC}"
        python -c "
import numpy as np
try:
    stats = np.load('models/sharky_evolution/sharky_$2_stats.npy', allow_pickle=True).item()
    print('📊 Detailed Statistics:')
    for key, value in stats.items():
        print(f'  {key}: {value}')
    print()
    if 'tournaments_played' in stats and stats['tournaments_played'] > 0:
        print('🏆 Performance Summary:')
        print(f'  Training Complete: ✅ ({stats[\"total_timesteps\"]:,} timesteps)')
        print(f'  Tournaments Played: {stats[\"tournaments_played\"]}')
        print(f'  Average Placement: {stats[\"average_placement\"]:.1f}/18')
        print(f'  Win Rate: {stats[\"win_rate\"]:.1%}')
        if stats['win_rate'] >= 0.3:
            print('  Assessment: 🔥 Exceptional!')
        elif stats['win_rate'] >= 0.15:
            print('  Assessment: ⭐ Very Good')
        elif stats['win_rate'] >= 0.08:
            print('  Assessment: 👍 Good')
        elif stats['win_rate'] >= 0.03:
            print('  Assessment: 📈 Learning')
        else:
            print('  Assessment: 🤔 Needs More Training')
    else:
        print('⚠️  No tournament evaluation data available')
        print('   Run: ./sharky.sh evaluate $2')
except FileNotFoundError:
    print('❌ Stats file not found for version $2')
    print('   Make sure the model is trained first')
except Exception as e:
    print(f'❌ Error loading stats: {e}')
"
        ;;
    
    "train-all")
        echo -e "${GREEN}🦈 Training all Sharky versions (1.0.0-1.0.9)...${NC}"
        python sharky_evolution_runner.py train-all 1.0
        ;;
    
    "tournament")
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo -e "${RED}❌ Error: Need at least 2 versions for tournament${NC}"
            echo "Usage: $0 tournament <version1> <version2> [version3...]"
            exit 1
        fi
        shift # Remove 'tournament' from arguments
        echo -e "${GREEN}🏟️ Running tournament between versions: $*${NC}"
        python sharky_evolution_runner.py tournament "$@"
        ;;
    
    "status")
        show_status
        ;;
    
    "help"|"--help"|"-h"|"")
        show_help
        ;;
    
    *)
        echo -e "${RED}❌ Error: Unknown command '$1'${NC}"
        echo
        show_help
        exit 1
        ;;
esac
