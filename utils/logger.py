import logging

logger = logging.getLogger("poker-ai")
logger.setLevel(logging.DEBUG)

# Console handler with formatter
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
console_handler.setFormatter(formatter)

# Avoid duplicate handlers when re-imported in notebooks or scripts
if not logger.hasHandlers():
    logger.addHandler(console_handler)
