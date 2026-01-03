import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

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

def avg_word_length(tokens):
    if not tokens:
        return 0.0
    return sum(len(word) for word in tokens) / len(tokens)

df["avg_word_length"] = df["tokens"].apply(avg_word_length)

print(df[["lyrics_title", "word_count", "avg_word_length"]].head())

sia = SentimentIntensityAnalyzer()

def sentiment_score(text):
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
    return sia.polarity_scores(text)["compound"]

df["sentiment_score"] = df["lyrics"].apply(sentiment_score)

print(df[["lyrics_title", "sentiment_score"]].head())

FIRST_PERSON = {"i", "me", "my", "mine", "im", "ive", "id"}

def first_person_pronoun_ratio(tokens):
    if not tokens:
        return 0.0
    count = sum(1 for t in tokens if t in FIRST_PERSON)
    return count / len(tokens)

df["first_person_pronoun_ratio"] = df["tokens"].apply(first_person_pronoun_ratio)

print(df[["lyrics_title", "first_person_pronoun_ratio"]].head())

EXPLICIT_WORDS = {
    "fuck", "fucking", "fucked", "fuckin",
    "shit", "shitty",
    "bitch", "bitches",
    "ass", "asshole",
    "dick", "pussy",
    "nigga", "niggas", # common in rap lyrics
    "hoe", "hoes"
}

def explicit_word_ratio(tokens):
    if not tokens:
        return 0.0
    count = sum(1 for t in tokens if t in EXPLICIT_WORDS)
    return count / len(tokens)

df["explicit_word_ratio"] = df["tokens"].apply(explicit_word_ratio)

print(df[["lyrics_title", "explicit_word_ratio"]].head())

import matplotlib.pyplot as plt

plt.figure()
plt.hist(df["track_views_num"], bins=30)
plt.xlabel("Raw Track Views")
plt.ylabel("Frequency")
plt.title("Distribution of Raw Track Views")
plt.show()

plt.figure()
plt.hist(df["log_track_views"], bins=30)
plt.xlabel("Log Track Views")
plt.ylabel("Frequency")
plt.title("Distribution of Log-Transformed Track Views")
plt.show()

# Raw track views show a heavily right-skewed distribution, while the log-transformed values are more symmetric and suitable for analysis.

plt.figure()
plt.scatter(df["word_count"], df["log_track_views"], alpha=0.6)
plt.xlabel("Word Count")
plt.ylabel("Log Track Views")
plt.title("Word Count vs Log-Transformed Track Views")
plt.show()

# Word count shows no strong relationship with log-transformed track views, indicating that song popularity varies widely across both shorter and longer lyrics.

plt.figure()
plt.scatter(df["sentiment_score"], df["log_track_views"], alpha=0.6)
plt.xlabel("Sentiment Score (VADER)")
plt.ylabel("Log Track Views")
plt.title("Sentiment vs Log-Transformed Track Views")
plt.show()
