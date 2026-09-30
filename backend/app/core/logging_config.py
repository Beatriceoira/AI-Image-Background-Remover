import logging

# Define the logging configuration
class LoggingConfig:
    # Define the logging level
    LEVEL = logging.INFO

    # Define the logging function
    def __init__(self):
        # Initialize the logging configuration
        logging.basicConfig(level=self.LEVEL)