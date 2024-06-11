import os
from pathlib import Path
import logging

# Set up logging configuration to display the time and message for each log entry
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# Define the name of the project
project_name = "mlProject"

# List of files to be created as part of the project structure
list_of_files = [
    ".github/workflows/.gitkeep",  # Placeholder file to ensure the directory is tracked by Git
    f"src/{project_name}/__init__.py",  # Initialization file for the src/mlProject package
    f"src/{project_name}/components/__init__.py",  # Initialization file for the components sub-package
    f"src/{project_name}/utils/__init__.py",  # Initialization file for the utils sub-package
    f"src/{project_name}/utils/common.py",  # Common utility functions will be placed here
    f"src/{project_name}/config/__init__.py",  # Initialization file for the config sub-package
    f"src/{project_name}/config/configuration.py",  # Configuration settings for the project
    f"src/{project_name}/pipeline/__init__.py",  # Initialization file for the pipeline sub-package
    f"src/{project_name}/entity/__init__.py",  # Initialization file for the entity sub-package
    f"src/{project_name}/entity/config_entity.py",  # Configuration entities for the project
    f"src/{project_name}/constants/__init__.py",  # Initialization file for the constants sub-package
    "config/config.yaml",  # Configuration file in YAML format
    "params.yaml",  # Parameters file in YAML format
    "schema.yaml",  # Schema definition file in YAML format
    "main.py",  # Main script to run the project
    "app.py",  # Script for the web application (if any)
    "Dockerfile",  # Docker configuration file
    "requirements.txt",  # List of dependencies for the project
    "setup.py",  # Script for setting up the project as a package
    "research/trials.ipynb",  # Jupyter notebook for research and experimentation
    "templates/index.html",  # HTML template for the web application
    "test.py"  # Script for running tests
]

# Loop through each file in the list
for filepath in list_of_files:
    filepath = Path(filepath)  # Convert the filepath string to a Path object for better manipulation
    filedir, filename = os.path.split(filepath)  # Split the path into directory and filename components

    # If the directory part of the path is not empty, create the directory
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)  # Create the directory if it doesn't exist
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    # If the file does not exist or is empty, create an empty file
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass  # Create an empty file
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")  # Log that the file already exists
