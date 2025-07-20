# poker-ai/engine/hand_evaluator.py

from collections import Counter
from itertools import combinations

# Map ranks to numeric values as per poker convention
RANK_MAP = {
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'T': 10,
    'J': 11,
    'Q': 12,
    'K': 13,
    'A': 14,
}

def card_rank(card):
    """Return the poker rank of a card as an int 2-14."""
    return RANK_MAP[card.rank]

def evaluate_hand(cards):
    """Evaluate the best 5-card poker hand from 7 cards."""
    best_rank = None
    best_five = None

    for five_cards in combinations(cards, 5):
        rank, ranked_five = hand_rank(five_cards)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_five = ranked_five

    return best_rank, best_five

def hand_rank(cards):
    """Return (rank_tuple, cards) for a 5-card hand."""
    ranks = sorted([card_rank(card) for card in cards], reverse=True)
    suits = [card.suit for card in cards]

    rank_counts = Counter(ranks)
    counts = sorted(rank_counts.values(), reverse=True)
    sorted_ranks = sorted(rank_counts.keys(), key=lambda r: (rank_counts[r], r), reverse=True)

    def get_cards_by_ranks(target_ranks):
        result = []
        for rank in target_ranks:
            result.extend([c for c in cards if card_rank(c) == rank])
        return result[:5]

    flush = None
    flush_cards = []
    for suit in set(suits):
        suited_cards = [card for card in cards if card.suit == suit]
        if len(suited_cards) >= 5:
            flush = sorted(suited_cards, key=card_rank, reverse=True)
            break

    def straight(ranks_list):
        ranks_set = set(ranks_list)
        for high in range(14, 5 - 1, -1):
            needed = set(range(high, high - 5, -1))
            if needed.issubset(ranks_set):
                return high
        if {14, 2, 3, 4, 5}.issubset(ranks_set):
            return 5
        return None

    # Straight flush
    if flush:
        flush_ranks = [card_rank(c) for c in flush]
        sf_high = straight(flush_ranks)
        if sf_high is not None:
            straight_flush_cards = []
            needed = list(range(sf_high, sf_high - 5, -1)) if sf_high != 5 else [5, 4, 3, 2, 14]
            for r in needed:
                for c in flush:
                    if card_rank(c) == r and c not in straight_flush_cards:
                        straight_flush_cards.append(c)
                        break
            return (8, sf_high), straight_flush_cards

    # Four of a kind
    if counts[0] == 4:
        four_rank = sorted_ranks[0]
        kicker = max(r for r in ranks if r != four_rank)
        cards_out = get_cards_by_ranks([four_rank, kicker])
        return (7, four_rank, kicker), cards_out

    # Full house
    if counts[0] == 3 and counts[1] >= 2:
        three_rank = sorted_ranks[0]
        pair_rank = sorted_ranks[1]
        cards_out = get_cards_by_ranks([three_rank, pair_rank])
        return (6, three_rank, pair_rank), cards_out

    # Flush
    if flush:
        return (5, *[card_rank(c) for c in flush[:5]]), flush[:5]

    # Straight
    straight_high = straight(ranks)
    if straight_high is not None:
        needed = list(range(straight_high, straight_high - 5, -1)) if straight_high != 5 else [5, 4, 3, 2, 14]
        straight_cards = []
        used = set()
        for r in needed:
            for c in cards:
                if card_rank(c) == r and c not in used:
                    straight_cards.append(c)
                    used.add(c)
                    break
        return (4, straight_high), straight_cards

    # Three of a kind
    if counts[0] == 3:
        three_rank = sorted_ranks[0]
        kickers = sorted([r for r in ranks if r != three_rank], reverse=True)[:2]
        cards_out = get_cards_by_ranks([three_rank] + kickers)
        return (3, three_rank, *kickers), cards_out

    # Two pair
    if counts[0] == 2 and counts[1] == 2:
        high_pair = sorted_ranks[0]
        low_pair = sorted_ranks[1]
        kicker = max(r for r in ranks if r != high_pair and r != low_pair)
        cards_out = get_cards_by_ranks([high_pair, low_pair, kicker])
        return (2, high_pair, low_pair, kicker), cards_out

    # One pair
    if counts[0] == 2:
        pair_rank = sorted_ranks[0]
        kickers = sorted([r for r in ranks if r != pair_rank], reverse=True)[:3]
        cards_out = get_cards_by_ranks([pair_rank] + kickers)
        return (1, pair_rank, *kickers), cards_out

    # High card
    return (0, *ranks[:5]), sorted(cards, key=card_rank, reverse=True)[:5]
