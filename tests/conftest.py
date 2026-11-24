"""Shared test setup: make the Django API importable and start Django once."""
import os
import sys
from pathlib import Path

import django

API_DIR = Path(__file__).resolve().parent.parent / "api"
sys.path.insert(0, str(API_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
os.environ.setdefault("DJANGO_ALLOWED_HOSTS", "testserver")  # host name the test client uses
django.setup()
