import os
import re
import numpy as np
from underthesea import word_tokenize

stop_word = set()
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "res/vietnamese-stopwords.txt"), "r", encoding="utf-8") as f:
    for line in f:
        word = line.strip()
        if word:   
            stop_word.add(word)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", '', text)
    tokens = word_tokenize(text)
    tokens = [word.replace(' ', '_') for word in tokens if word not in stop_word]
    return ' '.join(tokens)

def normalize(scores):
    scores = np.array(scores, dtype=np.float32)
    mean = scores.mean()
    std = scores.std()

    lower = mean - 3 * std
    upper = mean + 3 * std

    if upper == lower:
        return np.ones_like(scores) * 0.5

    normalized = (scores - lower) / (upper - lower)
    normalized = np.clip(normalized, 0, 1)
    return normalized.tolist()