import requests
from config import OPENWEATHERMAP_API_KEY
from typing import Dict

def fetch_weather_by_city(city: str, units: str = "metric") -> Dict:
    if not OPENWEATHERMAP_API_KEY:
        raise ValueError("Set OPENWEATHERMAP_API_KEY in env.")
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": OPENWEATHERMAP_API_KEY, "units": units}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

# Example output normalization:
def normalize_weather(data: Dict) -> str:
    name = data.get("name")
    weather = data.get("weather", [{}])[0].get("description", "")
    main = data.get("main", {})
    temp = main.get("temp")
    humidity = main.get("humidity")
    wind = data.get("wind", {}).get("speed")
    return f"Weather for {name}: {weather}. Temp: {temp}°C, Humidity: {humidity}%, Wind speed: {wind} m/s."
