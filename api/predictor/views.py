"""HTTP endpoints. Each view only checks input and returns JSON; services.py does the ML work."""
from rest_framework.response import Response
from rest_framework.views import APIView

from . import services
from .serializers import CarInputSerializer


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

    def post(self, request):
        serializer = CarInputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)  # bad input -> 400 with clear messages
        return Response(services.predict_price(serializer.validated_data))
