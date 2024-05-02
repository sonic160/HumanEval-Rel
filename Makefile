# Define Python interpreter
PYTHON = python

# Define project name
PROJECT_NAME = HUMANEVAL-REL

# Define source directory
SRC_DIR = src

# Define main Python file
MAIN_FILE = main.py

# Define requirements file
REQUIREMENTS_FILE = requirements.txt

# Define test directory
TEST_DIR = test.py

# Define virtual environment directory
VENV_DIR = venv

# Define targets
.PHONY: all all_test install run test clean

all: install run 

all_test: install run test

install:
	# Create virtual environment
	$(PYTHON) -m venv $(VENV_DIR)
	# Activate virtual environment and install requirements
	. $(VENV_DIR)/Scripts/Activate && \
	$(PYTHON) -m pip install --upgrade pip && \
	$(PYTHON) -m pip install -r $(REQUIREMENTS_FILE)

run:
	# Run main Python file
	. $(VENV_DIR)/Scripts/Activate && \
	$(PYTHON) $(SRC_DIR)/$(MAIN_FILE)

test:
	# Run tests
	. $(VENV_DIR)/Scripts/Activate && \
	$(PYTHON) -m pytest $(SRC_DIR)/$(TEST_DIR)

clean:
	# Clean up virtual environment
	rm -rf $(VENV_DIR)