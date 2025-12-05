# frontend/streamlit_app.py
import uuid
import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000"


def get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]


def init_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "city" not in st.session_state:
        st.session_state["city"] = ""


def main():
    st.set_page_config(page_title="Travel Guide Chatbot", page_icon="🧳")
    st.title("Travel Guide Chatbot")

    init_state()
    session_id = get_session_id()

    if not st.session_state["messages"]:
        intro = (
            "Hi, I'm your travel guide.\n\n"
            "First, tell me **which city** you're planning to visit "
            "(for example: Delhi, Paris, Bangkok)."
        )
        st.session_state["messages"].append(
            {"role": "assistant", "content": intro}
        )

    st.sidebar.header("Trip Settings")

    city = st.session_state.get("city", "")
    if city:
        st.sidebar.markdown(f"**Current city:** {city}")

    mode_label = st.sidebar.radio(
        "Planning mode",
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

    st.sidebar.subheader("Advanced")
    use_web = st.sidebar.checkbox("Use live web info", value=True)
    use_weather = st.sidebar.checkbox("Use live weather (OpenWeather)", value=True)
    use_events = st.sidebar.checkbox("Use local events (SerpAPI)", value=True)

    st.sidebar.caption(f"Session ID: `{session_id}`")

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type your reply or question...")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        if not st.session_state["city"]:
            city_name = user_input.strip()
            st.session_state["city"] = city_name

            confirm = (
                f"Great, we'll plan for **{city_name}**.\n\n"
                "Now you can:\n"
                "- Ask anything about this city, or\n"
                "- Use the sidebar to pick 1-day or multi-day itinerary mode, "
                "then describe your preferences (budget, pace, dates, etc.)."
            )
            st.session_state["messages"].append(
                {"role": "assistant", "content": confirm}
            )
            with st.chat_message("assistant"):
                st.markdown(confirm)
            return

        city = st.session_state["city"]
        try:
            resp = requests.post(
                f"{BACKEND_URL}/chat",
                json={
                    "session_id": session_id,
                    "message": user_input,
                    "city": city,
                    "mode": mode_value,
                    "use_web": use_web,
                    "use_weather": use_weather,
                    "use_events": use_events,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            bot_reply = data["reply"]
        except Exception as e:
            bot_reply = f"Error talking to backend: {e}"

        st.session_state["messages"].append(
            {"role": "assistant", "content": bot_reply}
        )
        with st.chat_message("assistant"):
            st.markdown(bot_reply)


if __name__ == "__main__":
    main()
