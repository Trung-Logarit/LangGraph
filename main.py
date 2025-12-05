import os
import json
from typing import TypedDict, Optional, List

import requests
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import StateGraph, END

genai.configure(api_key=os.environ["GEMINI_API_KEY"])


class AgentState(TypedDict):
    user_input: str
    intent: str
    expression: Optional[str]
    location: Optional[str]
    tool_result: Optional[str]
    final_answer: Optional[str]
    conversation_history: List[dict]


def router_node(state: AgentState) -> AgentState:
    user_text = state["user_input"]
    history = state["conversation_history"]

    history_text = ""
    if history:
        history_text = "Conversation so far:\n"
        for msg in history[-6:]:
            history_text += f"{msg['role']}: {msg['content']}\n"
        history_text += "\n"

    prompt = f"""
You are an intent classifier.

Decide:
- "calculator" → if the user wants math calculated. Extract the expression.
- "weather" → if the user asks about weather. Extract the location.
- "chat" → otherwise.

Output only JSON:
{{
  "intent": "...",
  "expression": "...",
  "location": "..."
}}

{history_text}
Current message: {user_text}

JSON:
"""

    model = genai.GenerativeModel("models/gemini-2.5-flash")
    result = model.generate_content(prompt)
    text = result.text.strip()

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(text)
        intent = data.get("intent", "chat")
        expr = data.get("expression", "") or None
        loc = data.get("location", "") or None
    except Exception:
        intent = "chat"
        expr = None
        loc = None

    return {
        "user_input": user_text,
        "intent": intent,
        "expression": expr,
        "location": loc,
        "tool_result": None,
        "final_answer": None,
        "conversation_history": history,
    }

def calculator_node(state: AgentState) -> AgentState:
    expr = state["expression"]

    if not expr:
        return {**state, "tool_result": None}

    try:
        value = eval(expr, {"__builtins__": {}}, {})
        result = str(value)
    except Exception as e:
        result = f"ERROR: {e}"

    return {**state, "tool_result": result}


def weather_node(state: AgentState) -> AgentState:
    loc = state["location"]
    if not loc:
        return {**state, "tool_result": "ERROR: No location provided."}

    url_geo = f"https://geocoding-api.open-meteo.com/v1/search?name={loc}&count=1"

    try:
        resp_geo = requests.get(url_geo, timeout=10)
        geo = resp_geo.json()

        if "results" not in geo or not geo["results"]:
            return {**state, "tool_result": f"ERROR: Cannot find '{loc}'"}

        latitude = geo["results"][0]["latitude"]
        longitude = geo["results"][0]["longitude"]


        url_weather = (
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude={latitude}&longitude={longitude}&current_weather=true"
        )

        resp_weather = requests.get(url_weather, timeout=10)
        weather = resp_weather.json()

        tool_result = json.dumps(
            {
                "location": geo["results"][0],
                "current_weather": weather.get("current_weather", {}),
            },
            ensure_ascii=False,
        )

    except Exception as e:
        tool_result = f"ERROR: weather API failed → {e}"

    return {**state, "tool_result": tool_result}


# ---------- CHAT NODE ----------
def chat_llm_node(state: AgentState) -> AgentState:
    history = state["conversation_history"]

    history_text = ""
    if history:
        for msg in history[-8:]:
            history_text += f"{msg['role']}: {msg['content']}\n"
        history_text += "\n"

    prompt = (
        "You are a helpful AI assistant.\n"
        + history_text +
        f"User: {state['user_input']}\nAssistant:"
    )

    model = genai.GenerativeModel("models/gemini-2.5-flash")
    answer = model.generate_content(prompt).text

    new_history = history + [
        {"role": "user", "content": state["user_input"]},
        {"role": "assistant", "content": answer},
    ]

    return {**state, "final_answer": answer, "conversation_history": new_history}


# ---------- FINAL NODE ----------
def final_node(state: AgentState) -> AgentState:
    history = state["conversation_history"]

    if state["intent"] == "chat":
        return {**state}

    if state["intent"] == "calculator":
        prompt = (
            f"User question: {state['user_input']}\n"
            f"Math result: {state['tool_result']}\n"
            "Explain this in 1–2 simple sentences."
        )

    elif state["intent"] == "weather":
        prompt = (
            f"Weather raw data: {state['tool_result']}\n"
            "Summarize the current weather in 2–3 sentences."
        )
    else:
        return {**state}

    model = genai.GenerativeModel("models/gemini-2.5-flash")
    answer = model.generate_content(prompt).text

    new_history = history + [
        {"role": "user", "content": state["user_input"]},
        {"role": "assistant", "content": answer},
    ]

    return {**state, "final_answer": answer, "conversation_history": new_history}


# ---------- BUILD GRAPH ----------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("calculator", calculator_node)
    graph.add_node("weather", weather_node)
    graph.add_node("chat_llm", chat_llm_node)
    graph.add_node("final", final_node)

    graph.set_entry_point("router")

    def route_next(state: AgentState):
        if state["intent"] == "calculator":
            return "calculator"
        if state["intent"] == "weather":
            return "weather"
        return "chat_llm"

    graph.add_conditional_edges("router", route_next, {
        "calculator": "calculator",
        "weather": "weather",
        "chat_llm": "chat_llm"
    })

    graph.add_edge("calculator", "final")
    graph.add_edge("weather", "final")
    graph.add_edge("chat_llm", "final")
    graph.add_edge("final", END)

    return graph.compile()


graph_app = build_graph()


# ---------- FASTAPI ----------
class ChatRequest(BaseModel):
    user_input: str
    conversation_history: List[dict] = []


class ChatResponse(BaseModel):
    answer: str
    intent: str
    tool_result: Optional[str]
    conversation_history: List[dict]


app = FastAPI()

@app.get("/")
def root():
    return {"status": "LangGraph Agent running ✔"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    state: AgentState = {
        "user_input": req.user_input,
        "intent": "",
        "expression": None,
        "location": None,
        "tool_result": None,
        "final_answer": None,
        "conversation_history": req.conversation_history,
    }

    new_state = graph_app.invoke(state)

    return ChatResponse(
        answer=new_state["final_answer"],
        intent=new_state["intent"],
        tool_result=new_state["tool_result"],
        conversation_history=new_state["conversation_history"],
    )
