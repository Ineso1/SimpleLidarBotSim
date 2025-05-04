#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
RESET='\033[0m'

CHECK_MARK="✔️"
CROSS_MARK="❌"
WARNING="⚠️"
INFO="ℹ️"
ROBOT="🤖"

spin() {
    local pid=$!
    local spinstr='|/-\\'
    local i=0
    echo -n ' '
    while kill -0 $pid 2>/dev/null; do
        i=$(( (i+1) %4 ))
        echo -n "${spinstr:$i:1}" "\b"
        sleep 0.1
    done
    echo " "
}

# Check if the virtual environment exists, if not, create it
echo -e "${CYAN}$ROBOT Checking for virtual environment...${RESET}"
if [ ! -d "env" ]; then
  echo -e "${YELLOW}$WARNING Creating virtual environment...${RESET}"
  python3 -m venv env
  echo -e "${GREEN}$CHECK_MARK Virtual environment created!${RESET}"
else
  echo -e "${GREEN}$CHECK_MARK Virtual environment already exists!${RESET}"
fi

# Activate the virtual environment
echo -e "${CYAN}$INFO Activating virtual environment...${RESET}"
source env/bin/activate

# Install dependencies
echo -e "${CYAN}$INFO Installing dependencies...${RESET}"
pip install -r requirements.txt & spin

# Check if pip install was successful
if [ $? -eq 0 ]; then
  echo -e "${GREEN}$CHECK_MARK Dependencies installed successfully!${RESET}"
else
  echo -e "${RED}$CROSS_MARK Failed to install dependencies.${RESET}"
  exit 1
fi

# Start the project 
echo -e "${CYAN}$INFO Starting the project...${RESET}"
python main.py & spin

# Check if Python script was successful
if [ $? -eq 0 ]; then
  echo -e "${GREEN}$CHECK_MARK Project started successfully!${RESET}"
else
  echo -e "${RED}$CROSS_MARK There was an error starting the project.${RESET}"
  exit 1
fi
