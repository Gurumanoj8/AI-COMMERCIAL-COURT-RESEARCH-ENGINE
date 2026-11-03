"""
Vercel Serverless Function entry point.
Exposes the Flask WSGI application for @vercel/python runtime.
"""
import sys
from pathlib import Path

# Add project root directory to Python path so app, config, and services are discoverable
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import app

# Vercel's serverless runtime invokes the WSGI application instance 'app'
