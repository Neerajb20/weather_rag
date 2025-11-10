import pytest
from src.nodes.weather_node import fetch_weather_by_city
from unittest.mock import patch

@patch("src.nodes.weather_node.requests.get")
def test_fetch_weather_ok(mock_get):
    class Dummy:
        def raise_for_status(self): pass
        def json(self): return {"name":"Mumbai","weather":[{"description":"clear sky"}],"main":{"temp":30,"humidity":70},"wind":{"speed":3.2}}
    mock_get.return_value = Dummy()
    data = fetch_weather_by_city("Mumbai")
    assert data["name"] == "Mumbai"
