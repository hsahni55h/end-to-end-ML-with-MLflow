import setuptools

# Read the contents of the README file to use as the long description for the package
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Define the current version of the package
__version__ = "0.0.0"

# Define constants for the repository and author details
REPO_NAME = "end-to-end-ML-with-MLflow"
AUTHOR_USER_NAME = "hsahni55h"
SRC_REPO = "mlProject"
AUTHOR_EMAIL = "h.sahni1998@gmail.com"

# Setup configuration for the package
setuptools.setup(
    name=SRC_REPO,  # Name of the package
    version=__version__,  # Version of the package
    author=AUTHOR_USER_NAME,  # Author's GitHub username
    author_email=AUTHOR_EMAIL,  # Author's email address
    description="A small python package for ml app",  # Short description of the package
    long_description=long_description,  # Long description read from the README file
    long_description_content="text/markdown",  # Format of the long description
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",  # URL of the project's GitHub repository
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",  # URL for the issue tracker
    },
    package_dir={"": "src"},  # Source directory for the package
    packages=setuptools.find_packages(where="src")  # Automatically find packages in the source directory
)

