"""Top-level URL map: the website at /, the API under /api/."""
from django.urls import include, path

from predictor.views import HomePageView

urlpatterns = [
    path("", HomePageView.as_view(), name="home"),
    path("api/", include("predictor.urls")),
]
