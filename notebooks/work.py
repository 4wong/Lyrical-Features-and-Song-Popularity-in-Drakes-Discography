import pandas as pd
import numpy as np

# Load the dataset from the data/ folder
df = pd.read_csv("data/drake_data.csv")

def parse_views(x):
    if pd.isna(x):
        return np.nan
    
    # Convert to string and strip spaces
    s = str(x).strip()
    if s == "":
        return np.nan

    # Handle K / M suffixes
    multiplier = 1
    if s[-1] in ["K", "k"]:
        multiplier = 1_000
        s = s[:-1]
    elif s[-1] in ["M", "m"]:
        multiplier = 1_000_000
        s = s[:-1]

    # Remove commas just in case
    s = s.replace(",", "")

    try:
        return float(s) * multiplier
    except ValueError:
        # Anything weird goes to NaN
        return np.nan

# Create numeric views column
df["track_views_num"] = df["track_views"].apply(parse_views)

# Drop rows where we couldn't parse views
df = df.dropna(subset=["track_views_num"])

# Create log-transformed views (base 10)
df["log_track_views"] = np.log10(df["track_views_num"])

print(df[["track_views", "track_views_num", "log_track_views"]].head())

import re

def preprocess_lyrics(text):
    if not isinstance(text, str):
        return []
    
    # lowercase
    text = text.lower()
    
    # remove punctuation & numbers
    text = re.sub(r"[^a-z\s]", "", text)
    
    # split into words
    tokens = text.split()
    
    return tokens

def is_valid_lyrics(tokens, min_words=100):
    return len(tokens) >= min_words

def unique_word_ratio(tokens):
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)

df["tokens"] = df["lyrics"].apply(preprocess_lyrics) 

# Filtering out snippets
df = df[df["tokens"].apply(is_valid_lyrics)].reset_index(drop=True)

df["word_count"] = df["tokens"].apply(len)
df["unique_word_ratio"] = df["tokens"].apply(unique_word_ratio)

print(df[["lyrics_title", "word_count", "unique_word_ratio"]].head())

