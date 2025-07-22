from typing import NamedTuple, Optional

class ActionValidationError(ValueError):
    pass

class RaiseValidationResult(NamedTuple):
    is_all_in: bool
    raise_amount: int
    amount_to_put_in: int
    reason: Optional[str] = None

def validate_raise(*, raise_to, player_stack, to_call, current_bet, min_raise, big_blind, player_current_bet) -> RaiseValidationResult:
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

    Returns:
    - RaiseValidationResult: namedtuple with is_all_in, raise_amount, amount_to_put_in, reason

    Raises ActionValidationError if invalid.

    Notes:
    - All-in raises are allowed even if below min_raise, as long as they are above current_bet.
    - All numeric arguments must be non-negative integers.
    """

    # Defensive: type and value checks
    for name, val in [
        ("raise_to", raise_to), ("player_stack", player_stack), ("to_call", to_call),
        ("current_bet", current_bet), ("min_raise", min_raise), ("big_blind", big_blind), ("player_current_bet", player_current_bet)
    ]:
        if not isinstance(val, int):
            raise ActionValidationError(f"{name} must be an integer.")
        if val < 0:
            raise ActionValidationError(f"{name} must be positive.")

    if player_stack == 0:
        raise ActionValidationError("Player has no chips left to bet.")

    if raise_to <= 0:
        raise ActionValidationError("raise_to must be positive.")

    amount_to_put_in = raise_to - player_current_bet

    if amount_to_put_in > player_stack:
        raise ActionValidationError(f"Invalid raise: player only has {player_stack} chips.")

    if raise_to <= player_current_bet:
        raise ActionValidationError("Raise must be greater than player's current bet.")

    # All-in detection
    is_all_in = amount_to_put_in == player_stack

    # All-in that is less than a call is not allowed
    if is_all_in and amount_to_put_in < to_call:
        raise ActionValidationError("All-in is not enough to call the current bet.")

    # All-in raise: must be above current_bet
    if is_all_in:
        if raise_to <= current_bet:
            raise ActionValidationError("All-in must be a raise above the current bet.")
        return RaiseValidationResult(is_all_in=True, raise_amount=raise_to - current_bet, amount_to_put_in=amount_to_put_in)

    # Opening bet (to_call == 0)
    if to_call == 0:
        if raise_to < big_blind:
            raise ActionValidationError(f"Opening bet must be at least the big blind ({big_blind}).")
    else:
        # Normal raise: must be at least min_raise
        raise_amount = raise_to - current_bet
        if raise_amount < min_raise:
            raise ActionValidationError(f"Must raise by at least {min_raise} chips (big blind or last raise).")

    # Defensive: player_current_bet > current_bet is not allowed
    if player_current_bet > current_bet:
        raise ActionValidationError("player_current_bet cannot be greater than current_bet.")

    return RaiseValidationResult(is_all_in=False, raise_amount=raise_to - current_bet, amount_to_put_in=amount_to_put_in)

def validate_call(*, player_stack, to_call):
    """
    Validate a call action in No-Limit Texas Hold'em.

    Parameters:
    - player_stack: chips player has (not counting what's already in the pot)
    - to_call: chips needed to call the current bet

    Returns:
    - dict with keys: is_all_in (bool), call_amount (int)

    Raises ActionValidationError if invalid.
    """
    if not isinstance(player_stack, int) or not isinstance(to_call, int):
        raise ActionValidationError("player_stack and to_call must be integers.")
    if player_stack < 0 or to_call < 0:
        raise ActionValidationError("player_stack and to_call must be non-negative.")

    if player_stack == 0:
        raise ActionValidationError("Player has no chips left to call.")

    # If player has enough chips, normal call
    if player_stack >= to_call:
        return {"is_all_in": False, "call_amount": to_call}
    # If player does not have enough, it's an all-in call for less
    elif player_stack < to_call:
        return {"is_all_in": True, "call_amount": player_stack}

def validate_check(*, to_call):
    """
    Validate a check action in No-Limit Texas Hold'em.

    Parameters:
    - to_call: chips needed to call the current bet

    Returns:
    - dict with key: can_check (bool)

    Raises ActionValidationError if invalid.
    """
    if not isinstance(to_call, int):
        raise ActionValidationError("to_call must be an integer.")
    if to_call < 0:
        raise ActionValidationError("to_call must be non-negative.")

    if to_call == 0:
        return {"can_check": True}
    else:
        raise ActionValidationError("Cannot check when there is a bet to call.")

def validate_fold(*, in_hand, to_call):
    """
    Validate a fold action in No-Limit Texas Hold'em.

    Parameters:
    - in_hand: bool, whether the player is currently in the hand
    - to_call: int, chips needed to call the current bet

    Returns:
    - dict with key: can_fold (bool)

    Raises ActionValidationError if invalid.
    """
    if not isinstance(in_hand, bool):
        raise ActionValidationError("in_hand must be a boolean.")
    if not isinstance(to_call, int):
        raise ActionValidationError("to_call must be an integer.")
    if to_call < 0:
        raise ActionValidationError("to_call must be non-negative.")

    if not in_hand:
        raise ActionValidationError("Cannot fold if player is not in hand.")
    if to_call == 0:
        raise ActionValidationError("Cannot fold when you can check (to_call == 0).")

    return {"can_fold": True}
