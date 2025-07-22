class RaiseValidationError(ValueError):
    pass

def validate_raise(*, raise_to, player_stack, to_call, current_bet, min_raise, big_blind, player_current_bet):
    """
    Validate raise in No-Limit Texas Hold'em.

    Parameters:
    - raise_to: total bet player wants to have in pot after raise
    - player_stack: chips player has *excluding* their current bet in pot
    - to_call: chips player must add to call current bet
    - current_bet: highest bet on table
    - min_raise: minimum raise increment
    - big_blind: big blind value
    - player_current_bet: player's current total bet in pot (before this action)

    Raises RaiseValidationError if invalid.
    """

    if raise_to <= 0:
        raise RaiseValidationError("Raise amount must be positive.")

    # How many chips the player must put in now to reach raise_to
    amount_to_put_in = raise_to - player_current_bet

    # Player cannot put in more chips than they have
    if amount_to_put_in > player_stack:
        raise RaiseValidationError(f"Invalid raise: player only has {player_stack} chips.")

    # Player cannot raise to an amount less or equal than their current bet
    if raise_to <= player_current_bet:
        raise RaiseValidationError("Raise must be greater than player's current bet.")

    # Detect if player is going all-in (putting all chips they have)
    is_all_in = amount_to_put_in == player_stack

    # If player is all-in, allow any raise_to above current_bet (even if less than min_raise)
    if is_all_in:
        # All-in must still be a raise (not just a call)
        if raise_to <= current_bet:
            raise RaiseValidationError("All-in must be a raise above the current bet.")
        return

    # Handle case when to_call == 0 (player is big blind or first to act with no bet)
    if to_call == 0:
        # Opening bet: must be at least big blind unless all-in (already handled above)
        if raise_to < big_blind:
            raise RaiseValidationError(f"Opening bet must be at least the big blind ({big_blind}).")
    else:
        # Normal raise: must raise by at least min_raise unless all-in (already handled above)
        raise_amount = raise_to - current_bet
        if raise_amount < min_raise:
            raise RaiseValidationError(f"Must raise by at least {min_raise} chips (big blind or last raise).")