#!/usr/bin/env python3
"""
Analysis of reward ratios in the poker AI system
"""

def analyze_reward_structure():
    print("=" * 80)
    print("POKER AI REWARD STRUCTURE ANALYSIS")
    print("=" * 80)
    
    print("\n1. PLACEMENT REWARDS (18-player tournament)")
    print("-" * 50)
    # Placement rewards from exponential system
    placement_rewards = [
        (1, 2000, "Winner - Tournament Victory"),
        (2, 1000, "Runner-up"),
        (3, 600, "3rd place"),
        (5, 208, "5th place"),
        (9, 108, "Final table bubble"),
        (10, 25, "Just missed final table"),
        (15, 25, "Middle elimination"),
        (18, 25, "First elimination")
    ]
    
    for place, reward, desc in placement_rewards:
        print(f"  Place {place:2d}: {reward:4d} - {desc}")
    
    print(f"\n  Ratio analysis:")
    print(f"    Winner vs 2nd: {2000/1000:.1f}x")
    print(f"    Winner vs 3rd: {2000/600:.1f}x")
    print(f"    Winner vs 9th: {2000/108:.1f}x")
    print(f"    2nd vs bubble: {1000/25:.1f}x")
    
    print("\n2. IMMEDIATE ACTION REWARDS")
    print("-" * 50)
    action_rewards = [
        ("Excellent bluff catch", 30, "Reading opponent correctly"),
        ("Disciplined fold vs strong", 25, "Avoiding trap"),
        ("Good fold vs strong hand", 20, "Smart laydown"),
        ("Hero call success", 20, "Calling big bluff"),
        ("Chip preservation 20%+", 20, "Growing stack significantly"),
        ("Bluffing success bonus", 15, "Successful aggression"),
        ("Good pot odds call", 15, "Mathematical decision"),
        ("Chip preservation 10%+", 10, "Growing stack moderately"),
        ("Short-handed aggression", 8, "Tournament aggression"),
        ("Good disciplined fold", 5, "Avoiding losses"),
        ("Failed aggression penalty", -3, "Poor timing"),
        ("Bad pot odds penalty", -10, "Poor mathematical decision"),
        ("10% stack loss", -20, "Moderate loss"),
        ("25% stack loss", -50, "Significant loss"),
        ("50%+ stack loss", -100, "Severe loss")
    ]
    
    for desc, reward, explanation in action_rewards:
        print(f"  {desc:25s}: {reward:4d} - {explanation}")
    
    print("\n3. ONGOING BONUSES")
    print("-" * 50)
    ongoing_bonuses = [
        ("Top 3 final table", 25, "per step when 9 left"),
        ("Top half final table", 15, "per step when 9 left"),
        ("Chip leader group", 15, "per step when 18 left"),
        ("General position", "0-10", "based on percentile"),
        ("Survival (short stack)", "0.5-2.0", "encourage survival"),
        ("Blind level 5", 33.5, "progression bonus"),
        ("Blind level 10", 94.9, "late tournament"),
        ("Blind level 15", 174.3, "endgame survival")
    ]
    
    for desc, reward, explanation in ongoing_bonuses:
        if isinstance(reward, (int, float)):
            print(f"  {desc:25s}: {reward:>8.1f} - {explanation}")
        else:
            print(f"  {desc:25s}: {reward:>8s} - {explanation}")
    
    print("\n4. STACK-DEPENDENT REWARDS")
    print("-" * 50)
    stack_rewards = [
        ("Win 100 chips", 30, "stack_change * 0.3, capped at 50"),
        ("Win 166 chips", 50, "maximum stack change reward"),
        ("Successful raise", "chip_change * 0.2", "capped at 25"),
        ("Successful call", "chip_change * 0.15", "capped at 20")
    ]
    
    for desc, reward, explanation in stack_rewards:
        if isinstance(reward, (int, float)):
            print(f"  {desc:25s}: {reward:>8.1f} - {explanation}")
        else:
            print(f"  {desc:25s}: {reward:>8s} - {explanation}")
    
    print("\n5. RATIO ANALYSIS & BALANCE ASSESSMENT")
    print("-" * 50)
    
    # Calculate ratios between different reward types
    winner_reward = 2000
    excellent_read = 30
    stack_change_max = 50
    severe_loss = -100
    progression_late = 174.3
    
    print(f"Key Ratios:")
    print(f"  Winner vs best single action: {winner_reward / excellent_read:.1f}x")
    print(f"  Winner vs max stack reward: {winner_reward / stack_change_max:.1f}x")
    print(f"  Winner vs severe loss: {winner_reward / abs(severe_loss):.1f}x")
    print(f"  Best action vs worst penalty: {excellent_read / abs(severe_loss):.1f}x")
    print(f"  Late progression vs best action: {progression_late / excellent_read:.1f}x")
    
    print(f"\nBalance Assessment:")
    print(f"  ✓ Tournament placement heavily favors top finishes (exponential)")
    print(f"  ✓ Single actions are meaningful but not overwhelming")
    print(f"  ✓ Severe losses have strong deterrent effect")
    print(f"  ✓ Progression bonuses scale with tournament difficulty")
    print(f"  ⚠ Late progression bonus might be too high vs single actions")
    
    print("\n6. POTENTIAL ISSUES")
    print("-" * 50)
    
    issues = [
        ("Progression bonus scaling", "174.3 at level 15 vs 30 for best read - might overshadow skill"),
        ("Position bonus frequency", "Up to 25 per step might accumulate too quickly"),
        ("Stack change caps", "Max 50 for wins might be too low for big pots"),
        ("Survival bonus range", "0.5-2.0 might be too small to matter"),
        ("Penalty severity", "-100 for 50% loss might be too harsh early game")
    ]
    
    for issue, explanation in issues:
        print(f"  • {issue}: {explanation}")
    
    print("\n7. RECOMMENDATIONS")
    print("-" * 50)
    
    recommendations = [
        ("Reduce progression bonus", "Scale by 0.5x: (level^1.5) * 1.5 instead of 3"),
        ("Increase survival bonus", "Scale 2-8 instead of 0.5-2.0 for short stacks"),
        ("Adjust position bonus", "Reduce frequency or magnitude for chip leaders"),
        ("Review stack caps", "Consider increasing stack change caps for big pots"),
        ("Add context to penalties", "Reduce penalties in early tournament phases")
    ]
    
    for rec, explanation in recommendations:
        print(f"  • {rec}: {explanation}")

if __name__ == "__main__":
    analyze_reward_structure()
