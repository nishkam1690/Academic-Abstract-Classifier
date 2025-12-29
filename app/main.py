from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware
from starlette.status import HTTP_302_FOUND


class AbstractRequest(BaseModel):
    abstract: str = Field(..., min_length=20, description="Raw abstract text in English.")


class PredictionResponse(BaseModel):
    predicted_field: str
    confidence: float
    top_k: List[dict]


def load_pipeline(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run `python src/train_classifier.py` first."
        )
    return joblib.load(model_path)


MODEL_PATH = Path("models/abstract_classifier.joblib")
pipeline = load_pipeline(MODEL_PATH)

app = FastAPI(
    title="Academic Abstract Classifier",
    description="Predict the research field for an academic abstract (ArXiv-inspired).",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Simple auth configuration ---
SECRET_KEY = "change-this-secret-key"
DEMO_USERNAME = "admin"
DEMO_PASSWORD = "Aditya123"

app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)


def _predict_top_k(abstract: str):
    proba = pipeline.predict_proba([abstract])[0]
    classes = pipeline.classes_
    top_sorted = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)
    top_k = [
        {"field": cls, "confidence": float(score)}
        for cls, score in top_sorted[: min(3, len(top_sorted))]
    ]
    return top_k


@app.get("/", response_class=HTMLResponse)
def landing_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
def landing_page_submit(request: Request, abstract: str = Form(...)):
    error = None
    result = None

    cleaned = abstract.strip()
    if len(cleaned) < 20:
        error = "Please provide an abstract with at least 20 characters."
    else:
        top_k = _predict_top_k(cleaned)
        result = {
            "predicted_field": top_k[0]["field"],
            "confidence": top_k[0]["confidence"],
            "top_k": top_k,
            "abstract": cleaned,
        }

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result, "error": error, "abstract": abstract},
    )


# ---------- Auth & Home pages ----------

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    # If already has a user, go straight to home
    if request.session.get("user"):
        return RedirectResponse(url="/home", status_code=HTTP_302_FOUND)

    # Show login form with username and password
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": None, "username": ""},
    )


@app.post("/login", response_class=HTMLResponse)
def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    # Check credentials
    if username == DEMO_USERNAME and password == DEMO_PASSWORD:
        request.session["user"] = username
        return RedirectResponse(url="/home", status_code=HTTP_302_FOUND)

    error = "Invalid username or password."
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": error, "username": username},
    )


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=HTTP_302_FOUND)


@app.get("/home", response_class=HTMLResponse)
def home_page(request: Request):
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/login", status_code=HTTP_302_FOUND)

    return templates.TemplateResponse("home.html", {"request": request, "user": user})


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/classify", response_model=PredictionResponse)
def classify_abstract(payload: AbstractRequest):
    if not payload.abstract.strip():
        raise HTTPException(status_code=400, detail="Abstract cannot be empty.")

    top_k = _predict_top_k(payload.abstract)

    return PredictionResponse(
        predicted_field=top_k[0]["field"],
        confidence=top_k[0]["confidence"],
        top_k=top_k,
    )

