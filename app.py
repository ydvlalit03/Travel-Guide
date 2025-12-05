import streamlit as st
import uuid
from chat_chain import chat_once   # direct import
import asyncio

# No backend URL needed anymore

def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]

def init_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "city" not in st.session_state:
        st.session_state["city"] = ""

st.set_page_config(page_title="Travel Guide Chatbot")
st.title("Travel Guide Chatbot")

init_state()
session_id = get_session_id()

if not st.session_state["messages"]:
    intro = (
        "Hi, I'm your travel guide.\n\n"
        "Tell me which city you want to travel to."
    )
    st.session_state["messages"].append({"role": "assistant", "content": intro})

# Sidebar
st.sidebar.header("Settings")

city = st.session_state["city"]
if city:
    st.sidebar.write("City Selected:", city)

mode = st.sidebar.selectbox(
    "Mode",
    ["chat", "day_plan", "multi_day"]
)

use_web = st.sidebar.checkbox("Use web data", True)
use_weather = st.sidebar.checkbox("Weather", True)
use_events = st.sidebar.checkbox("Events", True)

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state["messages"].append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # If no city, set first message as a city
    if not st.session_state["city"]:
        st.session_state["city"] = user_input.strip()
        bot_msg = f"Great. City set to **{user_input}**. Ask something or request itinerary."
    else:
        bot_msg = asyncio.run(chat_once(
            session_id=session_id,
            city=st.session_state["city"],
            user_message=user_input,
            mode=mode,
            use_web=use_web,
            use_weather=use_weather,
            use_events=use_events,
        ))

    st.session_state["messages"].append({"role":"assistant","content":bot_msg})
    with st.chat_message("assistant"):
        st.write(bot_msg)
