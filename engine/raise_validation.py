# poker-ai/engine/raise_validation.py
"""
Validate a raise action.

Parameters:
- current_bet (int): current highest bet on the table
- last_raise_amount (int): the amount of the last raise (minimum raise size)
- player_current_bet (int): how much the player has already put in the pot this betting round
- raise_to (int): the total amount the player wants to have put in after raising
- player_stack (int): how many chips the player has remaining (excluding current_bet)

Raises:
- ValueError with descriptive message if the raise is invalid.
"""

def validate_raise(current_bet, last_raise_amount, player_current_bet, raise_to, player_stack):
    """
    Validate that a player's raise_to amount is legal under standard No-Limit Hold'em rules.
    """

    # Total the player wants to bet (raise_to) can't exceed current_bet + player_stack
    max_total_bet = player_current_bet + player_stack
    if raise_to > max_total_bet:
        raise ValueError(
            f"Invalid raise: player only has {player_stack} chips, max total bet is {max_total_bet}, tried {raise_to}"
        )

    if raise_to <= player_current_bet:
        raise ValueError(
            f"Raise to {raise_to} must be greater than player's current bet {player_current_bet}."
        )

    raise_by = raise_to - current_bet

    if raise_by < last_raise_amount:
        # Allow it only if it's a full all-in
        total_required = raise_to - player_current_bet
        if total_required != player_stack:
            raise ValueError(
                f"Invalid raise: must raise by at least {last_raise_amount}, tried {raise_by}"
            )

