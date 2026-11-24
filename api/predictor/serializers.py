"""Check and tidy the JSON a user sends before it reaches the model."""
from datetime import date

from rest_framework import serializers

from src.data_cleaning import MAX_ENGINE_CC, MAX_MILEAGE, MIN_YEAR

from .services import get_options


def _match(value: str, allowed: list[str], field: str) -> str:
    """Return the allowed value that matches `value`, ignoring upper/lower case."""
    lookup = {option.lower(): option for option in allowed}
    try:
        return lookup[value.strip().lower()]
    except KeyError:
        raise serializers.ValidationError({field: f"Unknown {field} '{value}'."})


class CarInputSerializer(serializers.Serializer):
    make = serializers.CharField(max_length=50)
    model = serializers.CharField(max_length=60)
    year = serializers.IntegerField(min_value=MIN_YEAR, max_value=date.today().year + 1)
    mileage = serializers.IntegerField(min_value=0, max_value=MAX_MILEAGE)
    engine_cc = serializers.IntegerField(min_value=0, max_value=MAX_ENGINE_CC)
    fuel = serializers.CharField(max_length=20)
    transmission = serializers.CharField(max_length=20)
    city = serializers.CharField(max_length=50, required=False, default="Other")

    def validate(self, data: dict) -> dict:
        """Accept 'honda' or 'HONDA', and check the model belongs to the make."""
        options = get_options()
        data["make"] = _match(data["make"], list(options["makes"]), "make")
        data["model"] = _match(data["model"], options["makes"][data["make"]], "model")
        data["fuel"] = _match(data["fuel"], options["fuels"], "fuel")
        if data["fuel"] != "Electric" and data["engine_cc"] == 0:
            raise serializers.ValidationError(
                {"engine_cc": "Engine size is required for non-electric cars."}
            )
        data["transmission"] = _match(
            data["transmission"], options["transmissions"], "transmission"
        )
        # Small or unknown cities were grouped as "Other" during training
        city_lookup = {c.lower(): c for c in options["cities"]}
        data["city"] = city_lookup.get(data["city"].strip().lower(), "Other")
        return data
