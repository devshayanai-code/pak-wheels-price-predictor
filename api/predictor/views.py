"""HTTP endpoints. Each view only checks input and returns JSON; services.py does the ML work."""
from django.views.generic import TemplateView
from rest_framework.response import Response
from rest_framework.views import APIView

from src.config import TEST_SIZE

from . import services
from .serializers import CarInputSerializer


class HomePageView(TemplateView):
    """GET / - the website visitors use; it calls the API endpoints below."""

    template_name = "predictor/index.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        metrics = services.get_metrics()
        listings = services.get_listings()
        context.update(
            n_listings=f"{len(listings):,}",
            n_train=f"{round(len(listings) * (1 - TEST_SIZE)):,}",
            n_test=f"{len(listings) - round(len(listings) * (1 - TEST_SIZE)):,}",
            n_cities=listings.loc[listings["city"] != "Other", "city"].nunique(),
            error_percent=round(metrics["test"]["test_mape_%"]),
            model_name=metrics["best_model"].replace("_", " ").title(),
        )
        return context


class HealthView(APIView):
    """GET /api/health/ - lets the host (and a keep-alive ping) check the API is up."""

    def get(self, request):
        return Response({"status": "ok", "model": services.get_metrics()["best_model"]})


class OptionsView(APIView):
    """GET /api/options/ - valid makes, models, fuels, transmissions and cities."""

    def get(self, request):
        return Response(services.get_options())


class PredictView(APIView):
    """POST /api/predict/ - send a car as JSON, get back its estimated price."""

    throttle_scope = "predict"  # at most 60 predictions per minute per visitor

    def post(self, request):
        serializer = CarInputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)  # bad input -> 400 with clear messages
        return Response(services.predict_price(serializer.validated_data))
