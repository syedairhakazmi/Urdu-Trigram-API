import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import random

MODEL_PATH = os.getenv ("MODEL_PATH", "./trigram_model.json")
MAX_TOKENS  = int (os.getenv ("MAX_TOKENS", 150))
TEMPERATURE = float (os.getenv ("TEMPERATURE", 0.85))

app = FastAPI (title = "Urdu Trigram Generator API")

# Enable CORS FIRST - before loading model or defining routes
app.add_middleware (
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

print (f"Loading model from {MODEL_PATH} ...")
with open (MODEL_PATH, encoding = "utf-8") as f:
    _raw = json.load (f)

UNIGRAM: dict = _raw ["unigram"]           # { word: count }
TRIGRAM: dict = _raw ["trigram"]           # { "w1|||w2": { next_word: prob } }
META:    dict = _raw ["meta"]

TOTAL_TOKENS = META ["total_tokens"]
VOCAB_SIZE   = META ["vocab_size"]

print (f"  Unigram entries : {len (UNIGRAM):,}")
print (f"  Trigram contexts: {len (TRIGRAM):,}")
print ("Model ready.")
print ()

def _make_key (w1, w2):
    """Match the key format used by phase3_trigram_model.py"""
    return f"{w1}|||{w2}"

def _unigram_dist ():
    """Fallback: uniform distribution over unigram counts."""
    total = sum (UNIGRAM.values ())
    result = {}
    total = sum (UNIGRAM.values ())
    for w, c in UNIGRAM.items ():
        result [w] = c / total
    return result

def sample_next (word1, word2):
    key = _make_key (word1, word2)

    if key in TRIGRAM:
        dist = TRIGRAM [key]
    else:
        dist = _unigram_dist ()

    words   = list (dist.keys ())
    # Apply temperature scaling
    weights = []
    for p in dist.values ():
        weights.append (p ** (1.0 / TEMPERATURE))

    return random.choices (words, weights = weights, k = 1)[0]

def generate_text (prefix, max_tokens):
    tokens = prefix.strip ().split ()
    if len (tokens) < 2:
        raise HTTPException (
            status_code = 400,
            detail = "Prefix must contain at least 2 Urdu words separated by spaces."
        )

    w1, w2 = tokens [-2], tokens [-1]

    for _ in range (max_tokens):
        next_word = sample_next (w1, w2)

        # Stop at end-of-text special token
        if next_word in ("<EOT>", "\uE002"):
            break

        tokens.append (next_word)
        w1, w2 = w2, next_word

    return " ".join (tokens)

# Request / Response schemas 
class GenerateRequest (BaseModel):
    prefix: str
    max_tokens: int | None = None

class GenerateResponse (BaseModel):
    generated_text: str

class HealthResponse (BaseModel):
    status: str
    model_loaded: bool
    trigram_contexts: int
    vocab_size: int

# Routes
@app.post ("/generate", response_model = GenerateResponse)
def generate (req: GenerateRequest):
    if not req.prefix.strip ():
        raise HTTPException (status_code = 400, detail = "Prefix cannot be empty.")
    max_tokens = req.max_tokens or MAX_TOKENS
    text = generate_text (req.prefix, max_tokens)
    return GenerateResponse (generated_text = text)

@app.get ("/health")
def health ():
    return HealthResponse (
        status = "ok",
        model_loaded = True,
        trigram_contexts = len (TRIGRAM),
        vocab_size = VOCAB_SIZE,
    )

if __name__ == "__main__":
    port = int (os.environ.get ("PORT", 8000))

    print ()
    print (f" FastAPI running on port {port}")
    print ()

    uvicorn.run (
        app,
        host = "0.0.0.0",
        port = port,
        reload = False,
        log_level = "info"
    )
