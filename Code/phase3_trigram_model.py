# Phase 3: Trigram Language Model with Interpolation

import json
import random
import os
from collections import defaultdict, Counter

TOKENIZED_DIR = r"C:\Users\Syeda Irha Kazmi\OneDrive\Desktop\Sem_06\NLP\Assignment 1\urdu_stories_tokenized\urdu_stories_tokenized"
OUTPUT_MODEL = "trigram_model.json"

LAMBDA3 = 0.70
LAMBDA2 = 0.20
LAMBDA1 = 0.10

def load_tokens (directory):

    tokens = []

    files = os.listdir (directory)
    files.sort ()

    for name in files:
        if name.endswith (".txt"):
            path = os.path.join (directory, name)

            with open (path, encoding="utf-8", errors="ignore") as f:
                text = f.read ()
                parts = text.split ()

                for token in parts:
                    tokens.append (token)

    print ("Loaded", len (tokens), "tokens")
    return tokens

def build_count_tables (tokens):

    unigram = Counter (tokens)

    bigram_count = defaultdict (Counter)
    trigram_count = defaultdict (Counter)

    for i in range (len (tokens)-1):
        w1 = tokens [i]
        w2 = tokens [i+1]
        bigram_count [w1][w2] += 1

    for i in range (len (tokens)-2):
        w1 = tokens [i]
        w2 = tokens [i+1]
        w3 = tokens [i+2]
        trigram_count [(w1,w2)][w3] += 1

    print ("Unigrams:", len (unigram))
    print ("Bigram contexts:", len (bigram_count))
    print ("Trigram contexts:", len (trigram_count))

    return unigram, bigram_count, trigram_count

def build_interpolated_model (unigram, bigram_count, trigram_count):

    total_tokens = sum (unigram.values ())
    vocab_size = len (unigram)

    def p_unigram (w):
        return (unigram [w] + 1) / (total_tokens + vocab_size)

    def p_bigram (w, prev):
        total = sum (bigram_count [prev].values ())
        return (bigram_count [prev].get (w,0) + 1) / (total + vocab_size)

    def p_trigram (w, w1, w2):
        total = sum (trigram_count [(w1,w2)].values ())
        return (trigram_count [(w1,w2)].get (w,0) + 1) / (total + vocab_size)

    model = {}

    for context in trigram_count:
        w1 = context [0]
        w2 = context [1]

        next_words = trigram_count [context]

        dist = {}

        for w3 in next_words:
            prob = (
                LAMBDA3 * p_trigram (w3,w1,w2)
                + LAMBDA2 * p_bigram (w3,w2)
                + LAMBDA1 * p_unigram (w3)
            )
            dist [w3] = prob

        total = sum (dist.values ())

        norm = {}
        for word in dist:
            norm [word] = dist [word] / total

        model [context] = norm

    print ("Model built with", len (model), "contexts")
    return model, total_tokens, vocab_size

def save_model (model, unigram, bigram_count, total_tokens, vocab_size, path):

    data = {}
    data ["meta"] = {
        "total_tokens": total_tokens,
        "vocab_size": vocab_size,
        "lambda1": LAMBDA1,
        "lambda2": LAMBDA2,
        "lambda3": LAMBDA3
    }

    unigram_section = {}
    for word in unigram:
        unigram_section [word] = unigram [word]
    data ["unigram"] = unigram_section

    bigram_section = {}
    for w1 in bigram_count:
        inner = {}
        for w2 in bigram_count [w1]:
            inner [w2] = bigram_count [w1][w2]
        bigram_section [w1] = inner
    data ["bigram"] = bigram_section

    trigram_section = {}
    for context in model:

        w1 = context [0]
        w2 = context [1]

        key = w1 + "|||"+ w2

        inner = {}
        for w3 in model [context]:
            inner [w3] = model [context][w3]

        trigram_section [key] = inner

    data ["trigram"] = trigram_section

    with open (path,"w",encoding = "utf-8") as f:
        json.dump (data, f, ensure_ascii = False,indent = 2)

    size = os.path.getsize (path) / (1024 * 1024)
    print ("Saved model:", path, "Size:", round (size,2),"MB")

def load_model (path):

    with open (path,encoding = "utf-8") as f:
        data = json.load (f)

    meta = data ["meta"]

    unigram = Counter ()
    for w in data ["unigram"]:
        unigram [w] = data ["unigram"][w]

    bigram = {}
    for w1 in data ["bigram"]:
        for w2 in data ["bigram"][w1]:
            bigram [(w1,w2)] = data ["bigram"][w1][w2]

    model = {}
    for context in data ["trigram"]:
        parts = context.split ("|||")
        model [(parts [0],parts [1])] = data ["trigram"][context]

    return model, unigram, bigram, meta

def sample_next (dist):

    words = []
    weights = []

    for w in dist:
        words.append (w)
        weights.append (dist [w])

    return random.choices (words,weights = weights,k = 1)[0]

def generate_text (model, unigram, max_len = 100):

    w1 = "<BOS>"
    w2 = "<BOS>"

    generated = []

    for i in range (max_len):
        context = (w1,w2)

        if context in model:
            dist = model [context]
        else:
            dist = {}
            total = sum (unigram.values ())
            for w in unigram:
                dist [w] = unigram [w] / total

        word = sample_next (dist)

        if word == "<EOT>":
            break

        generated.append (word)

        w1 = w2
        w2 = word

    return " ".join (generated)

if __name__ == "__main__":
    print ()
    print ("Loading tokens...")
    print ()
    tokens = load_tokens (TOKENIZED_DIR)

    print ()
    print ("Building counts...")
    print ()    
    unigram, bigram_count, trigram_count = build_count_tables (tokens)

    print ()
    print ("Building interpolated model...")
    print ()
    model, total_tokens, vocab_size = build_interpolated_model (
        unigram, bigram_count, trigram_count
    )

    print ()
    print ("Saving model...")
    print ()
    save_model (model, unigram, bigram_count, total_tokens, vocab_size, OUTPUT_MODEL)

    print ()
    print ("Generating text...")
    print ()
    model_loaded, unigram_loaded, bigram_loaded, meta = load_model (OUTPUT_MODEL)

    text = generate_text (model_loaded, unigram_loaded,50)

    print ()
    print ("Generated Text:")
    print ()
    print (text)
    print ()
