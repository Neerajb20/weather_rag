from src.nodes.decision_node import should_fetch_weather

def test_weather_question():
    assert should_fetch_weather("What's the weather in Mumbai today?") is True

def test_non_weather_question():
    assert should_fetch_weather("What does the PDF say about annulus diameter?") is False
