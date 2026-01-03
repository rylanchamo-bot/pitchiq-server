import os
import json
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# ===============================
# LOAD ENVIRONMENT VARIABLES
# ===============================
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

# ===============================
# OPENAI CLIENT
# ===============================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===============================
# FASTAPI APP
# ===============================
app = FastAPI()

# ===============================
# SIMPLE USER TRACKING (MVP)
# ===============================
USERS = {}  # user_id -> {used, reset, vip}

class PredictRequest(BaseModel):
    user_id: str
    homeTeam: str
    awayTeam: str

def get_user(user_id: str):
    if user_id not in USERS:
        USERS[user_id] = {
            "used": 0,
            "reset": datetime.utcnow(),
            "vip": False
        }
    return USERS[user_id]

def reset_if_needed(user):
    if datetime.utcnow() - user["reset"] >= timedelta(hours=24):
        user["used"] = 0
        user["reset"] = datetime.utcnow()

# ===============================
# HEALTH ENDPOINTS
# ===============================
@app.get("/")
def root():
    return {"message": "PitchIQ server running"}

@app.get("/health")
def health():
    return {"ok": True}

# ===============================
# PREDICTION ENDPOINT
# ===============================
@app.post("/predict")
def predict(req: PredictRequest):
    user = get_user(req.user_id)
    reset_if_needed(user)

    if not user["vip"] and user["used"] >= 2:
        raise HTTPException(
            status_code=429,
            detail="Free limit reached. Try again after 24 hours."
        )

    # ---------- PROMPT (JSON ONLY) ----------
    prompt = f"""
Return ONLY valid JSON with keys:
score_home, score_away,
home_win_prob, draw_prob, away_win_prob,
shots_home, shots_away,
shots_on_target_home, shots_on_target_away,
corners_home, corners_away,
cards_home, cards_away,
explanation.

Match:
Home: {req.homeTeam}
Away: {req.awayTeam}

Rules:
- probabilities sum to 1
- scores are integers 0–5
- explanation under 60 words
"""

    # ---------- OPENAI CALL ----------
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        )
    except Exception as e:
        print("\n===== OPENAI ERROR =====")
        print(e)
        print("=======================\n")
        raise HTTPException(status_code=500, detail=str(e))

    # ---------- CLEAN OUTPUT ----------
    text = (response.output_text or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()

    # ---------- PARSE JSON ----------
    try:
        data = json.loads(text)
    except Exception:
        print("\n===== JSON PARSE FAILED =====")
        print(text)
        print("============================\n")
        raise HTTPException(
            status_code=500,
            detail="AI output was not valid JSON"
        )

    if not user["vip"]:
        user["used"] += 1

    return data
