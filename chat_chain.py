# chat_chain.py
import os
from typing import Dict, Optional

from dotenv import load_dotenv
import requests

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory

import asyncio

load_dotenv()  # for local development; on Streamlit Cloud we use st.secrets via app.py

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# In-memory session history
_SESSION_STORE: Dict[str, BaseChatMessageHistory] = {}


def get_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = ChatMessageHistory()
    return _SESSION_STORE[session_id]


def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.3,
    )


def get_research_tool() -> TavilySearchResults:
    """
    Tavily tool for live web research (real-time info).
    Use for: forecast, openings, disruptions, general web info.
    """
    return TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        description=(
            "Use this for current, real-time travel information: openings, "
            "closures, disruptions, current events, quick forecast summaries, etc."
        ),
    )


SYSTEM_PROMPT = """
You are a focused TRAVEL GUIDE and SMART ITINERARY PLANNER.

Core role:
- Help users plan trips in specific cities (any country).
- Answer questions about the city, and create single-day or multi-day itineraries.
- You will receive tool-generated contexts like:
  - current weather
  - forecast + recommended places
  - local events
  - live web snippets

City handling:
- The app provides a `city` field. Assume all questions are about that city
  unless the user clearly switches to another one.
- If the city is missing or ambiguous, ask the user which city they mean.

Modes:
- mode="chat":
  - General Q&A about the city: neighborhoods, must-see places, food,
    safety, best areas to stay, how many days to spend, etc.
- mode="day_plan":
  - Create a detailed 1-day itinerary.
  - Structure chronologically: Morning, Late Morning, Lunch, Afternoon, Evening, Night.
  - Include approximate time windows and short travel hints between stops.
- mode="multi_day":
  - Create a multi-day itinerary (use 3–5 days if the user doesn’t specify).
  - Each day: 3–6 key activities in a logical route.
  - Balance sightseeing, local food, culture, and rest.

Tool contexts:
- `weather_context`:
  - Current weather from OpenWeatherMap (temp, feels-like, humidity, conditions, wind).
  - Use it to decide indoor vs outdoor activities and suggest what to wear or carry:
    - hot (>30°C): light clothes, water, sunglasses
    - cold (<15°C): jacket, layers
    - rain: umbrella/raincoat, indoor backup
    - windy: windbreaker
- `forecast_context`:
  - A short text combining weather forecast + suggested places to visit today.
  - Use this to time outdoor attractions and decide which sights fit best today.
- `events_context`:
  - Local events (title, date/time, venue, link).
  - Weave relevant events into the plan if timings fit the day.
  - Always include the event link when you mention it.
- `research_context`:
  - Live web search snippets (Tavily).
  - Use mainly when timings, closures, or very current info matters.

Style:
- Tone: friendly, like a local friend who knows the city well.
- Structure:
  - Use headings and bullet points for itineraries.
  - Use clear time ranges like "9:00–11:00".
- Safety & honesty:
  - If you are unsure or see no data, say so and avoid making up details.
  - Use approximate language for prices and schedules unless they appear explicitly.
- Emojis:
  - Use sparingly; focus on clarity and structure, not decoration.
"""


def _tavily_search(query: str) -> str:
    research_tool = getattr(chat_chain, "research_tool", None)
    if research_tool is None:
        return ""
    try:
        res = research_tool.invoke({"query": query})
    except Exception:
        return ""
    if isinstance(res, list):
        snippets = [r.get("content", "") for r in res if r.get("content")]
        return "\n\n".join(snippets)
    elif isinstance(res, str):
        return res
    return ""


def _get_research_context(query: str, city: str, use_web: bool) -> str:
    if not use_web:
        return ""
    search_q = f"{query} in {city}"
    return _tavily_search(search_q)


def _get_forecast_and_places_context(city: str, use_web: bool) -> str:
    """
    Use Tavily to get short forecast + key places to visit today.
    """
    if not use_web:
        return ""
    query = (
        f"Short weather forecast for today in {city} (°C) and 8-12 top places "
        f"to visit today, with a line or two about why they are good."
    )
    return _tavily_search(query)


def _get_weather_context(city: str, use_weather: bool) -> str:
    """
    Current weather via OpenWeatherMap, formatted as short text.
    """
    if not use_weather or not OPENWEATHER_API_KEY:
        return ""

    city = (city or "").strip()
    if not city:
        return ""

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return ""

    try:
        main = data.get("main", {})
        weather_list = data.get("weather", [])
        wind = data.get("wind", {})
        sys = data.get("sys", {})

        temp = main.get("temp")
        feels_like = main.get("feels_like")
        humidity = main.get("humidity")
        desc = weather_list[0]["description"] if weather_list else ""
        wind_speed = wind.get("speed")
        country = sys.get("country", "")

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
            return ""

        return " ".join(parts)
    except Exception:
        return ""


def _get_events_context(city: str, use_events: bool) -> str:
    """
    Local events via SerpAPI Google Events engine.
    """
    if not use_events or not SERPAPI_API_KEY:
        return ""

    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_events",
        "q": f"Events in {city}",
        "api_key": SERPAPI_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return ""

    events = data.get("events_results") or []
    if not events:
        return ""

    lines = []
    for ev in events[:8]:
        title = ev.get("title", "Event")
        start = ev.get("date", {}).get("start_date", "")
        when = ev.get("date", {}).get("when", "")
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

    if not lines:
        return ""

    return "Local events:\n" + "\n".join(lines)


def build_chain() -> RunnableWithMessageHistory:
    llm = get_llm()
    research_tool = get_research_tool()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "system",
                "Current city: {city}\n"
                "Mode: {mode}\n\n"
                "Live web research context (if any):\n{research_context}\n\n"
                "Forecast + places context (if any):\n{forecast_context}\n\n"
                "Weather context (if any):\n{weather_context}\n\n"
                "Events context (if any):\n{events_context}",
            ),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    chain_core = prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain_core,
        get_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="output",
    )

    chain_with_history.research_tool = research_tool
    return chain_with_history


chat_chain = build_chain()


async def chat_once(
    session_id: str,
    city: Optional[str],
    user_message: str,
    mode: str = "chat",
    use_web: bool = True,
    use_weather: bool = True,
    use_events: bool = True,
) -> str:
    """
    One chat turn for the travel bot (async).
    """
    city = (city or "").strip()
    if not city:
        return (
            "Before I can help, tell me which city you're planning to visit "
            "(for example: 'Jaipur', 'Paris', 'Bangkok')."
        )

    msg = user_message.strip()

    if mode == "day_plan":
        effective_input = (
            f"You are planning a single full day in {city}. "
            f"User preferences or constraints: {msg}. "
            "Create a realistic, enjoyable 1-day plan with clear time blocks."
        )
    elif mode == "multi_day":
        effective_input = (
            f"You are planning a multi-day trip in {city}. "
            f"User description / constraints: {msg}. "
            "If the user does not specify days, assume 3–5 days. "
            "Create a day-wise itinerary with balanced sightseeing, food, and rest."
        )
    else:
        effective_input = msg

    research_context = _get_research_context(effective_input, city, use_web=use_web)
    forecast_context = _get_forecast_and_places_context(city, use_web=use_web)
    weather_context = _get_weather_context(city, use_weather=use_weather)
    events_context = _get_events_context(city, use_events=use_events)

    result = await chat_chain.ainvoke(
        {
            "input": effective_input,
            "city": city,
            "mode": mode,
            "research_context": research_context,
            "forecast_context": forecast_context,
            "weather_context": weather_context,
            "events_context": events_context,
        },
        config={"configurable": {"session_id": session_id}},
    )

    if isinstance(result, AIMessage):
        return result.content
    return str(result)


def chat_once_sync(
    session_id: str,
    city: Optional[str],
    user_message: str,
    mode: str = "chat",
    use_web: bool = True,
    use_weather: bool = True,
    use_events: bool = True,
) -> str:
    """
    Synchronous wrapper for Streamlit.
    """
    return asyncio.run(
        chat_once(
            session_id=session_id,
            city=city,
            user_message=user_message,
            mode=mode,
            use_web=use_web,
            use_weather=use_weather,
            use_events=use_events,
        )
    )
