# LangGraph Tool-Calling Agent (FastAPI + Gemini)

This project implements a minimal **tool-calling AI agent** using:

- **Google Gemini (`gemini-2.5-flash`)**
- **LangGraph** for routing and state management
- **FastAPI** for API serving
- Built-in tools:
  - **Calculator**
  - **Weather lookup (Open-Meteo API)**
- Simple **conversation memory** (passed in request/response)

---

## 1. Requirements

Install dependencies:

```bash
pip install fastapi uvicorn google-generativeai langgraph requests pydantic
```

Set your Gemini API key:

```bash
export GEMINI_API_KEY="YOUR_KEY"
```

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_KEY"
```

---

## 2. Running the API

Start the FastAPI server:

```bash
uvicorn main:app --reload --port 8000
```

Open Swagger UI:

http://localhost:8000/docs

---

## 3. /chat Endpoint

### Method:  
`POST /chat`

### Request body:

```json
{
  "user_input": "What's the weather in Tokyo?",
  "conversation_history": []
}
```

### Response:

```json
{
  "answer": "...",
  "intent": "weather",
  "tool_result": "{...}",
  "conversation_history": [...]
}
```

---

## 4. How It Works

### **Router Node**
Classifies the intent using Gemini:
- `"calculator"`
- `"weather"`
- `"chat"`

### **Calculator Node**
Evaluates expressions using restricted Python `eval`.

### **Weather Node**
Uses Open-Meteo Geocoding + Weather API.

### **Chat LLM Node**
Generates a natural-language response using Gemini.

### **Final Node**
Summarizes tool results (weather/calculator) using Gemini.

---

## 5. Extending the Agent

To add a new tool:

1. Add new intent description to router prompt  
2. Add new fields to `AgentState`  
3. Implement a new LangGraph node  
4. Register it in the graph  
5. Update `route_next()`  
6. Connect it to the `final` node  

---

## 6. Notes

- Calculator uses Python `eval` → OK for demo, not recommended for production  
- Agent is stateless; client must always send `conversation_history`  
- Easy to extend for more tools (search, DB query, RAG, etc.)

---

End of README.

