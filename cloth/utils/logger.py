import logging



RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
PURPLE = "\033[35m"

class CustomFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: BLUE, # Blue
        logging.INFO: GREEN, # Green
        logging.WARNING: YELLOW, # Yellow
        logging.ERROR: RED, # Red
        logging.CRITICAL: PURPLE, # Purple
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno)
        levelname = record.levelname
        max_length = 18  # To match FastAPI alignment
        colored_levelname = f"{color}{levelname}{RESET}"
        padded_levelname = '{:<{width}}'.format(colored_levelname + ":", width=max_length)
        record.levelname = padded_levelname
        return super(CustomFormatter, self).format(record)

def get_logger(logger_name="cloth", log_level=logging.DEBUG):
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False  # Prevent log propagation to avoid double logging

    formatter = CustomFormatter("{levelname} {filename}/{funcName} -> {message}", style="{")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger
