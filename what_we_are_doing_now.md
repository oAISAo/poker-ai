NOW:
implementing handle_raise()



AFTER:

Once handle_raise() is implemented:

Catch the ValueError inside step() instead of handle_raise() if you want to separate UI/logic even more.

Let handle_raise() return a structured result if needed for logging/debugging.

Write a few unit tests for it (with mock players/states).

Later, do the same for call, check, and fold logic.

Would you like help scaffolding out the handle_raise() method now with proper integration of validate_raise()?