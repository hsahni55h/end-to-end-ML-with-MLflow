import os
import sys
import logging

# Define the format for log messages
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Define the directory and file path for log files
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")

# Create the log directory if it doesn't already exist
os.makedirs(log_dir, exist_ok=True)

# Set up the basic configuration for logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format=logging_str,  # Use the defined format for log messages
    handlers=[
        logging.FileHandler(log_filepath),  # Log messages to a file
        logging.StreamHandler(sys.stdout)  # Also output log messages to the console (stdout)
    ]
)

# Create a logger object with a specific name
logger = logging.getLogger("mlProjectLogger")

# Example usage:
#logger.info("This is an info message")
#logger.error("This is an error message")
