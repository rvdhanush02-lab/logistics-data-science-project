#!/usr/bin/env python
"""
Quick Start: Run the complete project
"""
import subprocess
import sys

print("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

print("\nRunning project evaluation...")
exec(open("evaluate.py").read())
