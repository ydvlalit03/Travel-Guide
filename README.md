Travel Guide Chatbot (FastAPI + Streamlit + Gemini + Weather + Events + Live Search)

This project is a full-stack AI-powered travel guide chatbot.
It helps users explore cities, plan trips, and receive itineraries based on weather, events, and live information.
The bot asks the city first, then responds conversationally.
It supports normal city chat, one-day plans, and multi-day itineraries.

Features

Conversational travel chatbot using Gemini Flash.

Live weather reports using OpenWeatherMap API.

Local events using SerpAPI Google Events.

Live search information using Tavily for real-time queries.

Per-session memory using ChatMessageHistory.

Three travel modes:

Chat about the city

One-day itinerary

Multi-day itinerary (default 3–5 days if not specified)

Frontend in Streamlit, backend in FastAPI.
