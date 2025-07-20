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
    assert [c.rank for c in best] == ["A", "K", "Q", "J", "9"]

def test_one_pair():
    cards = make_cards([("K", "♠"), ("K", "♣"), ("Q", "♦"), ("J", "♥"), ("9", "♠"), ("3", "♦"), ("2", "♣")])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 1  # One pair
    assert rank[1] == 13  # Pair of Kings
    assert [c.rank for c in best].count("K") == 2

def test_two_pair():
    cards = make_cards([("K", "♠"), ("K", "♣"), ("Q", "♦"), ("Q", "♥"), ("9", "♠"), ("3", "♦"), ("2", "♣")])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 2  # Two pair
    assert rank[1] == 13  # High pair Kings
    assert rank[2] == 12  # Low pair Queens
    assert [c.rank for c in best].count("K") == 2
    assert [c.rank for c in best].count("Q") == 2

def test_three_of_a_kind():
    cards = make_cards([("K", "♠"), ("K", "♣"), ("K", "♦"), ("J", "♥"), ("9", "♠"), ("3", "♦"), ("2", "♣")])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 3
    assert rank[1] == 13
    assert [c.rank for c in best].count("K") == 3

def test_straight_normal():
    cards = make_cards([("9", "♠"), ("8", "♣"), ("7", "♦"), ("6", "♥"), ("5", "♠"), ("3", "♦"), ("2", "♣")])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 4  # Straight
    assert rank[1] == 9   # Highest card 9

def test_straight_wheel():
    cards = make_cards([("A", "♠"), ("2", "♣"), ("3", "♦"), ("4", "♥"), ("5", "♠"), ("9", "♦"), ("7", "♣")])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 4  # Straight
    assert rank[1] == 5  # Highest card 5 for wheel straight

def test_flush():
    cards = make_cards([("A", "♠"), ("K", "♠"), ("Q", "♠"), ("J", "♠"), ("9", "♠"), ("3", "♦"), ("2", "♣")])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 5
    assert all(c.suit == "♠" for c in best)

def test_full_house():
    cards = make_cards([("K", "♠"), ("K", "♣"), ("K", "♦"), ("J", "♥"), ("J", "♠"), ("3", "♦"), ("2", "♣")])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 6
    assert rank[1] == 13  # Trips Kings
    assert rank[2] == 11  # Pair Jacks

def test_four_of_a_kind():
    cards = make_cards([("K", "♠"), ("K", "♣"), ("K", "♦"), ("K", "♥"), ("9", "♠"), ("3", "♦"), ("2", "♣")])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 7
    assert rank[1] == 13  # Quads Kings

def test_straight_flush():
    cards = make_cards([("9", "♠"), ("8", "♠"), ("7", "♠"), ("6", "♠"), ("5", "♠"), ("3", "♦"), ("2", "♣")])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 8
    assert rank[1] == 9

def test_best_hand_from_seven_cards():
    cards = make_cards([("9", "♠"), ("8", "♠"), ("7", "♠"), ("6", "♠"), ("5", "♠"), ("A", "♦"), ("A", "♣")])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 8  # Straight flush beats pair of aces
    assert rank[1] == 9

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
    assert all(c.suit == "♠" for c in best)
    assert "A" in [c.rank for c in best]

def test_multiple_flushes():
    cards = make_cards([
        ("A", "♠"), ("K", "♠"), ("Q", "♠"), ("J", "♠"), ("9", "♠"),
        ("A", "♥"), ("K", "♥"), ("Q", "♥"), ("J", "♥"), ("9", "♥"),
        ("2", "♦")
    ])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 5
    # Flush should be spades (highest suit flush)
    assert all(c.suit == "♠" for c in best)
    assert "A" in [c.rank for c in best]

def test_full_house_multiple_triplets():
    cards = make_cards([
        ("3", "♠"), ("3", "♣"), ("3", "♦"),
        ("5", "♠"), ("5", "♥"), ("5", "♦"),
        ("7", "♣")
    ])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 6  # Full house
    # Should pick highest trips (5s) and next trips as pair (3s)
    assert rank[1] == 5
    assert rank[2] == 3

def test_four_of_a_kind_kicker():
    cards = make_cards([
        ("7", "♠"), ("7", "♣"), ("7", "♦"), ("7", "♥"),
        ("A", "♠"), ("K", "♠"), ("Q", "♣")
    ])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 7
    assert rank[1] == 7
    # Kicker should be Ace highest
    assert rank[2] == 14

def test_straight_flush_wheel():
    cards = make_cards([
        ("A", "♠"), ("2", "♠"), ("3", "♠"), ("4", "♠"), ("5", "♠"),
        ("9", "♦"), ("7", "♣")
    ])
    rank, best = evaluate_hand(cards)
    assert rank[0] == 8
    assert rank[1] == 5  # Ace-low straight flush highest card is 5

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
    # Best flush cards are top 5 highest cards
    assert [c.rank for c in best] == ["A", "K", "Q", "J", "9"]
