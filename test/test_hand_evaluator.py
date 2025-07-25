import pytest
from engine.cards import Card
from engine.hand_evaluator import evaluate_hand

def make_cards(ranks_suits):
    """Helper: create Card objects from list of (rank, suit)."""
    return [Card(rank, suit) for rank, suit in ranks_suits]

def test_high_card():
    cards = make_cards([("A", "♠"), ("K", "♣"), ("Q", "♦"), ("J", "♥"), ("9", "♠"), ("3", "♦"), ("2", "♣")])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 0  # High card
    assert best == [14, 13, 12, 11, 9]

def test_one_pair():
    cards = make_cards([("K", "♠"), ("K", "♣"), ("Q", "♦"), ("J", "♥"), ("9", "♠"), ("3", "♦"), ("2", "♣")])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 1  # One pair
    assert rank[1] == 13  # Pair of Kings
    assert best.count(13) == 2

def test_two_pair():
    cards = make_cards([("K", "♠"), ("K", "♣"), ("Q", "♦"), ("Q", "♥"), ("9", "♠"), ("3", "♦"), ("2", "♣")])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 2  # Two pair
    assert rank[1] == 13  # High pair Kings
    assert rank[2] == 12  # Low pair Queens
    assert best.count(13) == 2
    assert best.count(12) == 2

def test_three_of_a_kind():
    cards = make_cards([("K", "♠"), ("K", "♣"), ("K", "♦"), ("J", "♥"), ("9", "♠"), ("3", "♦"), ("2", "♣")])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 3
    assert rank[1] == 13
    assert best.count(13) == 3

def test_straight_normal():
    cards = make_cards([("9", "♠"), ("8", "♣"), ("7", "♦"), ("6", "♥"), ("5", "♠"), ("3", "♦"), ("2", "♣")])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 4  # Straight
    assert rank[1] == 9   # Highest card 9
    assert best == [9, 8, 7, 6, 5]

def test_straight_wheel():
    cards = make_cards([("A", "♠"), ("2", "♣"), ("3", "♦"), ("4", "♥"), ("5", "♠"), ("9", "♦"), ("7", "♣")])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 4  # Straight
    assert rank[1] == 5  # Highest card 5 for wheel straight
    assert best == [5, 4, 3, 2, 14]

def test_flush():
    cards = make_cards([("A", "♠"), ("K", "♠"), ("Q", "♠"), ("J", "♠"), ("9", "♠"), ("3", "♦"), ("2", "♣")])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 5
    # All best cards should be spades
    spade_ranks = [card_rank(c) for c in cards if c.suit == "♠"]
    assert all(r in spade_ranks for r in best)
    assert best == sorted(spade_ranks, reverse=True)[:5]

def test_full_house():
    cards = make_cards([("K", "♠"), ("K", "♣"), ("K", "♦"), ("J", "♥"), ("J", "♠"), ("3", "♦"), ("2", "♣")])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 6
    assert rank[1] == 13  # Trips Kings
    assert rank[2] == 11  # Pair Jacks
    assert best.count(13) == 3
    assert best.count(11) == 2

def test_four_of_a_kind():
    cards = make_cards([("K", "♠"), ("K", "♣"), ("K", "♦"), ("K", "♥"), ("9", "♠"), ("3", "♦"), ("2", "♣")])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 7
    assert rank[1] == 13  # Quads Kings
    assert best.count(13) == 4

def test_straight_flush():
    cards = make_cards([("9", "♠"), ("8", "♠"), ("7", "♠"), ("6", "♠"), ("5", "♠"), ("3", "♦"), ("2", "♣")])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 8
    assert rank[1] == 9
    assert best == [9, 8, 7, 6, 5]

def test_best_hand_from_seven_cards():
    cards = make_cards([("9", "♠"), ("8", "♠"), ("7", "♠"), ("6", "♠"), ("5", "♠"), ("A", "♦"), ("A", "♣")])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 8  # Straight flush beats pair of aces
    assert rank[1] == 9
    assert best == [9, 8, 7, 6, 5]

def test_tie_breakers():
    # Compare two pair with different kickers
    cards1 = make_cards([("K", "♠"), ("K", "♣"), ("Q", "♦"), ("Q", "♥"), ("9", "♠"), ("3", "♦"), ("2", "♣")])
    cards2 = make_cards([("K", "♠"), ("K", "♣"), ("Q", "♦"), ("Q", "♥"), ("8", "♠"), ("3", "♦"), ("2", "♣")])
    rank1, best1 = evaluate_hand(cards1)
    rank2, best2 = evaluate_hand(cards2)
    assert rank1 > rank2

    # Compare one pair with different kickers
    cards3 = make_cards([("K", "♠"), ("K", "♣"), ("J", "♦"), ("9", "♥"), ("8", "♠"), ("3", "♦"), ("2", "♣")])
    cards4 = make_cards([("K", "♠"), ("K", "♣"), ("J", "♦"), ("7", "♥"), ("6", "♠"), ("3", "♦"), ("2", "♣")])
    rank3, best3 = evaluate_hand(cards3)
    rank4, best4 = evaluate_hand(cards4)
    assert rank3 > rank4

def test_ace_low_flush():
    cards = make_cards([
        ("A", "♠"), ("J", "♠"), ("9", "♠"), ("6", "♠"), ("3", "♠"),
        ("2", "♦"), ("4", "♣")
    ])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 5  # Flush
    spade_ranks = [card_rank(c) for c in cards if c.suit == "♠"]
    assert all(r in spade_ranks for r in best)
    assert 14 in best  # Ace present

def test_multiple_flushes():
    cards = make_cards([
        ("A", "♠"), ("K", "♠"), ("Q", "♠"), ("J", "♠"), ("9", "♠"),
        ("A", "♥"), ("K", "♥"), ("Q", "♥"), ("J", "♥"), ("9", "♥"),
        ("2", "♦")
    ])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 5
    spade_ranks = [card_rank(c) for c in cards if c.suit == "♠"]
    assert all(r in spade_ranks for r in best)
    assert 14 in best  # Ace present

def test_full_house_multiple_triplets():
    cards = make_cards([
        ("3", "♠"), ("3", "♣"), ("3", "♦"),
        ("5", "♠"), ("5", "♥"), ("5", "♦"),
        ("7", "♣")
    ])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 6  # Full house
    assert best.count(5) == 3
    assert best.count(3) == 2

def test_four_of_a_kind_kicker():
    cards = make_cards([
        ("7", "♠"), ("7", "♣"), ("7", "♦"), ("7", "♥"),
        ("A", "♠"), ("K", "♠"), ("Q", "♣")
    ])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 7
    assert rank[1] == 7
    assert best.count(7) == 4
    assert 14 in best  # Ace kicker

def test_straight_flush_wheel():
    cards = make_cards([
        ("A", "♠"), ("2", "♠"), ("3", "♠"), ("4", "♠"), ("5", "♠"),
        ("9", "♦"), ("7", "♣")
    ])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 8
    assert rank[1] == 5  # Ace-low straight flush highest card is 5
    assert best == [5, 4, 3, 2, 14]

def test_tie_breakers_kickers():
    cards1 = make_cards([
        ("K", "♠"), ("K", "♣"), ("J", "♦"), ("9", "♥"), ("8", "♠"), ("3", "♦"), ("2", "♣")
    ])
    cards2 = make_cards([
        ("K", "♠"), ("K", "♣"), ("J", "♦"), ("7", "♥"), ("6", "♠"), ("3", "♦"), ("2", "♣")
    ])
    rank1, _ = evaluate_hand(cards1)
    rank2, _ = evaluate_hand(cards2)
    assert rank1 > rank2

def test_flush_selects_top_five():
    cards = make_cards([
        ("A", "♠"), ("K", "♠"), ("Q", "♠"), ("J", "♠"), ("9", "♠"),
        ("7", "♠"), ("6", "♠")
    ])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 5
    assert best == [14, 13, 12, 11, 9]

# Helper for rank conversion
def card_rank(card):
    RANK_MAP = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
        'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
    }
    return RANK_MAP[card.rank]