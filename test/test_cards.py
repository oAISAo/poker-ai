from engine.cards import Card, Deck
from engine.game import PokerGame
from engine.player import Player
from utils.enums import GameMode
import sys
print(sys.path)

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

def test_deck_creation():
    deck = Deck()
    assert len(deck.cards) == 52

def test_deck_shuffle():
    deck1 = Deck()
    deck2 = Deck()
    deck2.shuffle()
    assert deck1.cards != deck2.cards

def test_deck_draw():
    deck = Deck()
    cards = deck.draw(5)
    assert len(cards) == 5
    assert len(deck.cards) == 47

def test_deck_reset():
    deck = Deck()
    deck.draw(5)
    deck.reset()
    assert len(deck.cards) == 52