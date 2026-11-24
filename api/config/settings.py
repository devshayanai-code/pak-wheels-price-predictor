"""Django settings for the price prediction API.

The API only serves predictions, so it needs no database, login or admin pages.
Secret values come from environment variables on the server (Render).
"""
import os
import sys
from pathlib import Path

API_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = API_DIR.parent

# Let the API import the ML code in src/ (e.g. src.config)
sys.path.insert(0, str(ROOT_DIR))

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-insecure-key-change-me")
# Safe by default: debug pages only when DJANGO_DEBUG=True is set on purpose
DEBUG = os.environ.get("DJANGO_DEBUG", "False") == "True"
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

INSTALLED_APPS = [
    "rest_framework",
    "predictor",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"

DATABASES = {}  # no database: the model and data are loaded from files

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_PARSER_CLASSES": ["rest_framework.parsers.JSONParser"],
    "DEFAULT_AUTHENTICATION_CLASSES": [],  # public demo API, no login
    "DEFAULT_PERMISSION_CLASSES": [],
    "UNAUTHENTICATED_USER": None,
    # Limit abuse of the free server: only views that set a throttle_scope are limited
    "DEFAULT_THROTTLE_CLASSES": ["rest_framework.throttling.ScopedRateThrottle"],
    "DEFAULT_THROTTLE_RATES": {"predict": "60/min"},
    # Render sits in front of the app as 1 proxy; trust only its visitor IP header
    "NUM_PROXIES": 1,
}

CACHES = {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}}

USE_TZ = True
TIME_ZONE = "Asia/Karachi"
