import os
import uuid
import requests
import streamlit as st
from datetime import datetime

from google import genai
from google.genai import types

# -------------------- CONFIG / KEYS --------------------

# On Streamlit Cloud these come from st.secrets
# Locally you can use .env or export env vars
def load_keys():
    google_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
    if not google_key:
        st.stop()
    model_name = st.secrets.get("GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    openweather_key = st.secrets.get("OPENWEATHER_API_KEY", os.getenv("OPENWEATHER_API_KEY"))
    serpapi_key = st.secrets.get("SERPAPI_API_KEY", os.getenv("SERPAPI_API_KEY"))
    return google_key, model_name, openweather_key, serpapi_key


GOOGLE_API_KEY, GEMINI_MODEL, OPENWEATHER_API_KEY, SERPAPI_API_KEY = load_keys()

client = genai.Client(api_key=GOOGLE_API_KEY)


# -------------------- UTILS: SESSION STATE --------------------

def get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]


def init_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "city" not in st.session_state:
        st.session_state["city"] = ""


# -------------------- REAL-TIME TOOLS --------------------

def get_weather_raw(city: str):
    """Get raw weather JSON from OpenWeatherMap."""
    if not OPENWEATHER_API_KEY:
        return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}"
        params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def format_weather_text(city: str, data: dict | None) -> str:
    """Turn raw JSON into a short human-friendly description."""
    if not data:
        return "No live weather data available."
    try:
        main = data.get("main", {})
        weather_list = data.get("weather", [])
        wind = data.get("wind", {})
        sys_info = data.get("sys", {})

        temp = main.get("temp")
        feels_like = main.get("feels_like")
        humidity = main.get("humidity")
        desc = weather_list[0]["description"] if weather_list else ""
        wind_speed = wind.get("speed")
        country = sys_info.get("country", "")

        parts = []
        if temp is not None and feels_like is not None:
            parts.append(
                f"Current temperature in {city}{', ' + country if country else ''}: "
                f"{temp:.1f}°C (feels like {feels_like:.1f}°C)."
            )
        if desc:
            parts.append(f"Conditions: {desc}.")
        if humidity is not None:
            parts.append(f"Humidity: {humidity}%")
        if wind_speed is not None:
            parts.append(f"Wind: {wind_speed} m/s")

        if not parts:
            return "Weather data could not be parsed."

        return " ".join(parts)
    except Exception:
        return "Weather data could not be parsed."


def get_local_events(city: str):
    """Fetch events from SerpAPI Google Events."""
    if not SERPAPI_API_KEY:
        return []

    url =f"https://serpapi.com/search.json?engine=google_events&q=Events in {city}&api_key={SERPAPI_API_KEY}"
    params = {
        "engine": "google_events",
        "q": f"Events in {city}",
        "api_key": SERPAPI_API_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    return data.get("events_results") or []


def format_events_text(events: list) -> str:
    if not events:
        return "No live events found."
    lines = []
    for ev in events[:8]:
        title = ev.get("title", "Event")
        when = ev.get("date", {}).get("when", "")
        start = ev.get("date", {}).get("start_date", "")
        venue = ev.get("venue", {}).get("name", "")
        link = ev.get("link", "")
        line = f"- {title}"
        if when:
            line += f" | When: {when}"
        elif start:
            line += f" | Date: {start}"
        if venue:
            line += f" | Venue: {venue}"
        if link:
            line += f" | Link: {link}"
        lines.append(line)
    return "\n".join(lines)


# -------------------- LLM CALL --------------------

SYSTEM_INSTRUCTIONS = """
You are a travel guide and itinerary planner.

Rules:
- The user is always planning a trip in the given CITY.
- You are given:
  - mode: "chat", "day_plan", or "multi_day"
  - weather_context: what the weather is like now
  - events_context: upcoming local events
  - conversation_history: previous user and assistant messages
  - user_query: the latest user message

Behaviors:
- If mode="chat":
  - Answer travel questions about the city: neighborhoods, key attractions,
    food, safety, areas to stay, best time to visit, etc.
- If mode="day_plan":
  - Create a realistic 1-day itinerary for the city.
  - Structure the day as Morning / Late Morning / Lunch / Afternoon / Evening / Night.
  - Include approximate time windows and short travel hints.
- If mode="multi_day":
  - Create a multi-day itinerary (3–5 days if user did not specify).
  - Each day list 3–6 main activities in a sensible route.
  - Balance sightseeing, food, and rest; do not overload days.

Use weather_context:
- If hot (>30°C) or very sunny: suggest lighter clothes, hydration, and
  avoid intense walking at midday.
- If cold (<15°C): suggest warm layers, jacket.
- If rain/storm: suggest indoor activities or backups, and mention umbrellas.
- If windy: hint at windbreaker or caution in exposed viewpoints.

Use events_context:
- If relevant events exist, weave them into the plan (with links) at appropriate times.
- Only include events that reasonably fit a day's schedule.

Style:
- Write like a friendly local who knows the city well.
- Use short paragraphs and bullet points for itineraries.
- Do not overuse emojis.
- If some information is uncertain, say so and suggest checking official sources.
"""


def build_llm_input(
    city: str,
    mode: str,
    weather_text: str,
    events_text: str,
    history: list[dict],
    user_message: str,
) -> str:
    """Build a single text prompt for Gemini containing all context."""
    history_lines = []
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            history_lines.append(f"User: {content}")
        elif role == "assistant":
            history_lines.append(f"Assistant: {content}")

    history_block = "\n".join(history_lines) if history_lines else "No previous messages."

    text = f"""
SYSTEM INSTRUCTIONS:
{SYSTEM_INSTRUCTIONS}

CITY:
{city}

MODE:
{mode}

WEATHER CONTEXT:
{weather_text}

EVENTS CONTEXT:
{events_text}

CONVERSATION HISTORY:
{history_block}

USER QUERY:
{user_message}

Now answer as the travel guide, following the rules above.
"""
    return text.strip()


def call_gemini(prompt_text: str) -> str:
    """Call Gemini with a single text prompt."""
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt_text,
        config=types.GenerateContentConfig(
            # keep it simple; system instructions already inside text
            temperature=0.4,
        ),
    )
    # New google-genai library returns candidates/parts
    try:
        return resp.candidates[0].content.parts[0].text
    except Exception:
        # fallback
        return str(resp)


# -------------------- STREAMLIT UI --------------------

def main():
    st.set_page_config(page_title="Travel Guide Chatbot")
    st.title("Travel Guide Chatbot")

    init_state()
    session_id = get_session_id()

    # Initial assistant message
    if not st.session_state["messages"]:
        intro = (
            "Hi, I am your travel guide.\n\n"
            "First, tell me which city you are planning to visit."
        )
        st.session_state["messages"].append(
            {"role": "assistant", "content": intro}
        )

    # Sidebar
    st.sidebar.header("Trip settings")

    city = st.session_state.get("city", "")
    if city:
        st.sidebar.markdown(f"Current city: **{city}**")

    mode_label = st.sidebar.radio(
        "Mode",
        options=[
            "Just chat about this city",
            "Plan a 1-day itinerary",
            "Plan a multi-day itinerary",
        ],
        index=0,
    )
    mode_map = {
        "Just chat about this city": "chat",
        "Plan a 1-day itinerary": "day_plan",
        "Plan a multi-day itinerary": "multi_day",
    }
    mode_value = mode_map[mode_label]

    use_weather = st.sidebar.checkbox("Use live weather", value=True)
    use_events = st.sidebar.checkbox("Use live events", value=True)

    st.sidebar.caption(f"Session ID: {session_id}")

    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_input = st.chat_input("Type your question or request...")

    if user_input:
        # Show user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # First message sets the city
        if not st.session_state["city"]:
            city_name = user_input.strip()
            st.session_state["city"] = city_name
            reply = (
                f"Great, we will plan for **{city_name}**.\n\n"
                "Now you can ask anything about this city, or choose 1-day / multi-day mode in the sidebar and ask for an itinerary."
            )
        else:
            city = st.session_state["city"]

            # Prepare live contexts
            weather_text = "Live weather disabled."
            if use_weather:
                weather_data = get_weather_raw(city)
                weather_text = format_weather_text(city, weather_data)

            events_text = "Events lookup disabled."
            if use_events:
                events_data = get_local_events(city)
                events_text = format_events_text(events_data)

            # Build prompt for Gemini
            prompt_text = build_llm_input(
                city=city,
                mode=mode_value,
                weather_text=weather_text,
                events_text=events_text,
                history=st.session_state["messages"][:-1],  # exclude current user msg
                user_message=user_input,
            )

            reply = call_gemini(prompt_text)

        # Show assistant reply
        st.session_state["messages"].append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)


if __name__ == "__main__":
    main()
