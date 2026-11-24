"""Unit tests: each cleaning step on tiny hand-made tables."""
import pandas as pd

from src.data_cleaning import (
    _split_slug,
    clean_engine,
    clean_mileage,
    clean_price,
    group_rare_cities,
    remove_duplicates,
)


def test_clean_price_converts_text_and_drops_missing():
    df = pd.DataFrame({"price": ["PKR 3,650,000", "Call for price", None]})
    result = clean_price(df)
    assert result["price"].tolist() == [3_650_000]


def test_clean_mileage_removes_commas_and_units():
    df = pd.DataFrame({"mileage": ["138,000 km"]})
    assert clean_mileage(df)["mileage"].tolist() == [138_000]


def test_clean_engine_gives_electric_cars_zero_cc():
    df = pd.DataFrame({"engine": ["1800 cc", "60.0 kWh"]})
    result = clean_engine(df)
    assert result["engine_cc"].tolist() == [1800, 0]
    assert "engine" not in result.columns


def test_remove_duplicates_keeps_one_row_per_url():
    df = pd.DataFrame({"url": ["a", "a", "b"], "category": ["x", "y", "x"]})
    assert len(remove_duplicates(df)) == 2


def test_split_slug_handles_one_and_two_word_makes():
    assert _split_slug("honda-civic") == ("Honda", "Civic")
    assert _split_slug("mercedes-benz-c-class") == ("Mercedes-Benz", "C Class")


def test_group_rare_cities_replaces_small_cities():
    df = pd.DataFrame({"city": ["Lahore"] * 3 + ["Kohat"]})
    result = group_rare_cities(df, min_count=2)
    assert result["city"].tolist() == ["Lahore"] * 3 + ["Other"]
