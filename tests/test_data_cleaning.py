"""Unit tests: each cleaning step on tiny hand-made tables."""
import pandas as pd

from src.data_cleaning import (
    _split_slug,
    clean_engine,
    clean_mileage,
    clean_price,
    group_rare_cities,
    remove_duplicates,
    remove_fake_mileage,
    remove_price_typos,
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


def test_remove_price_typos_drops_a_missing_zero():
    prices = [2_450_000, 2_400_000, 2_500_000, 2_380_000, 2_520_000, 244_000]
    df = pd.DataFrame({"make": "Suzuki", "model": "Wagon R", "year": 2019, "price": prices})
    assert remove_price_typos(df)["price"].min() == 2_380_000


def test_remove_price_typos_keeps_rare_cars():
    # Too few cars to judge what is normal, so nothing is dropped
    df = pd.DataFrame({"make": "Toyota", "model": "Supra", "year": 2020,
                       "price": [30_000_000, 2_000_000]})
    assert len(remove_price_typos(df)) == 2


def test_remove_fake_mileage_only_on_older_cars():
    df = pd.DataFrame({"year": [2015, 2023, 2015], "mileage": [1, 5, 80_000]})
    result = remove_fake_mileage(df)
    assert result["mileage"].tolist() == [5, 80_000]  # a new car can have 5 km
