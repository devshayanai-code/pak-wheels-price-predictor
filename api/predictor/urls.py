from django.urls import path

from .views import HealthView, OptionsView, PredictView

urlpatterns = [
    path("health/", HealthView.as_view(), name="health"),
    path("options/", OptionsView.as_view(), name="options"),
    path("predict/", PredictView.as_view(), name="predict"),
]
