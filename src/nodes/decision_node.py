from typing import Dict
import re

def should_fetch_weather(question: str) -> bool:
    # simplistic heuristics — will route questions matching weather terms
    weather_keywords = ["weather", "temperature", "rain", "sunny", "forecast", "wind", "humidity"]
    question_l = question.lower()
    # If question contains city name pattern like "in <city>" or "for <city>" and weather keywords
    if any(k in question_l for k in weather_keywords):
        return True
    # If explicit "weather" returns true
    if re.search(r"\bweather\b", question_l):
        return True
    return False

def decision_node(question: str) -> Dict:
    if should_fetch_weather(question):
        route = "weather"
    else:
        route = "pdf_rag"
    return {"route": route}
