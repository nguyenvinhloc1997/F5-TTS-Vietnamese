#!/bin/bash

# This script creates a Python virtual environment and installs all dependencies using uv

# Exit on any error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up F5-TTS-Vietnamese environment${NC}"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}uv is not installed. Please install it first:${NC}"
    echo "curl -fsSL https://install.uv.dev | sh"
    exit 1
fi

# Create a virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo -e "${GREEN}Creating Python virtual environment with Python 3.12...${NC}"
    uv venv --python 3.12 .venv
else
    echo -e "${YELLOW}Virtual environment already exists.${NC}"
fi

# Activate the virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source .venv/bin/activate

# Install dependencies
echo -e "${GREEN}Installing dependencies from uv.yaml...${NC}"
uv pip install -e .

echo -e "${GREEN}Environment setup complete!${NC}"
echo -e "${GREEN}To activate the environment, run:${NC}"
echo -e "${YELLOW}source .venv/bin/activate${NC}"
echo -e "${GREEN}To run the TTS inference script:${NC}"
echo -e "${YELLOW}./tts_infer.py${NC}" 