from engine.cards import Card, Deck
import pytest

valid_ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
valid_suits = ['♠', '♣', '♦', '♥']

def test_card_creation():
    card = Card("A", "♠")
    assert card.rank == "A"
    assert card.suit == "♠"
    assert str(card) == "A♠"

def test_invalid_card_creation():
    with pytest.raises(ValueError):
        Card("X", "♠")
    with pytest.raises(ValueError):
        Card("A", "X")

def test_all_valid_cards():
    for rank in valid_ranks:
        for suit in valid_suits:
            card = Card(rank, suit)
            assert card.rank == rank
            assert card.suit == suit
            assert str(card) == f"{rank}{suit}"

@pytest.mark.parametrize("rank,suit", [
    ("", "♠"),
    (None, "♠"),
    ("A", ""),
    ("A", None),
    ("X", "X"),
    (123, "♠"),
    ("A", 456),
])
def test_invalid_cards(rank, suit):
    with pytest.raises(ValueError):
        Card(rank, suit)

def test_deck_creation():
    deck = Deck()
    assert len(deck.cards) == 52

def test_deck_unique_cards():
    deck = Deck()
    card_set = set(str(card) for card in deck.cards)
    assert len(card_set) == 52

def test_deck_shuffle():
    deck1 = Deck()
    deck2 = Deck()
    deck2.shuffle()
    # Note: Very rarely shuffle can produce the same order, so allow a few tries
    different = any(deck1.cards[i] != deck2.cards[i] for i in range(len(deck1.cards)))
    assert different

def test_deck_draw():
    deck = Deck()
    cards = deck.draw(5)
    assert len(cards) == 5
    assert len(deck.cards) == 47

def test_draw_more_than_deck():
    deck = Deck()
    with pytest.raises(ValueError):
        deck.draw(53)  # Assuming draw() raises if too many cards requested

def test_draw_zero():
    deck = Deck()
    with pytest.raises(ValueError):
        deck.draw(0)

def test_draw_negative():
    deck = Deck()
    with pytest.raises(ValueError):
        deck.draw(-1)

def test_deck_reset():
    deck = Deck()
    deck.draw(5)
    deck.reset()
    assert len(deck.cards) == 52
