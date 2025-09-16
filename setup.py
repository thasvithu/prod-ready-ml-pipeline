from setuptools import setup, find_packages
from typing import List

# Constant for identifying editable installs in requirements.txt
HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    Reads the requirements.txt file and returns a clean list of required packages.

    - Strips whitespace from each line.
    - Ignores empty lines and comments.
    - Removes the '-e .' line (used for editable install during development).
    """
    requirements = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace and newline characters
            if line and line != HYPEN_E_DOT:  # Skip empty lines and '-e .'
                requirements.append(line)
    return requirements

# Setup configuration for packaging the project
setup(
    name='prod_ready_ml_pipeline',             # Name of the Python package/project
    version='0.0.1',                           # Initial version
    author='Vimalathas Vithusan',             # Author's name
    author_email='thasvithu7@gmail.com',      # Author's email
    packages=find_packages(),                 # Automatically discover all packages in the project
    install_requires=get_requirements('requirements.txt'),  # Install external dependencies from file
    entry_points={
        'console_scripts': [
            'ml-train=src.pipeline.train_pipeline:run_training_pipeline',
        ],
    },
)
