"""Top-level URL map: every endpoint lives under /api/."""
from django.urls import include, path

urlpatterns = [
    path("api/", include("predictor.urls")),
]
