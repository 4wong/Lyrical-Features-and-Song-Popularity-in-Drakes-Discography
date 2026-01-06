import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

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

def avg_word_length(tokens):
    if not tokens:
        return 0.0
    return sum(len(word) for word in tokens) / len(tokens)

df["avg_word_length"] = df["tokens"].apply(avg_word_length)

sia = SentimentIntensityAnalyzer()

def sentiment_score(text):
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
    return sia.polarity_scores(text)["compound"]

df["sentiment_score"] = df["lyrics"].apply(sentiment_score)

FIRST_PERSON = {"i", "me", "my", "mine", "im", "ive", "id"}

def first_person_pronoun_ratio(tokens):
    if not tokens:
        return 0.0
    count = sum(1 for t in tokens if t in FIRST_PERSON)
    return count / len(tokens)

df["first_person_pronoun_ratio"] = df["tokens"].apply(first_person_pronoun_ratio)


from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

def load_word_list(path: Path) -> set[str]:
    """Load a newline-delimited wordlist, ignoring blank lines and # comments."""
    with path.open("r", encoding="utf-8") as f:
        return {
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        }

EXPLICIT_WORDS_PATH = ROOT_DIR / "data" / "explicit_words.txt"
EXPLICIT_WORDS = load_word_list(EXPLICIT_WORDS_PATH)


def explicit_word_ratio(tokens):
    if not tokens:
        return 0.0
    count = sum(1 for t in tokens if t in EXPLICIT_WORDS)
    return count / len(tokens)

df["explicit_word_ratio"] = df["tokens"].apply(explicit_word_ratio)

print("Loaded explicit words:", len(EXPLICIT_WORDS))

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

# Highly popular songs appear across both strongly positive and strongly negative sentiment values, indicating that overall lyrical polarity is not a strong predictor of popularity.

# Filtering albums with more than 5 tracks as albums with very few songs make boxplots noisy
album_counts = df["album"].value_counts()
valid_albums = album_counts[album_counts >= 5].index
df_album = df[df["album"].isin(valid_albums)]
 
plt.figure(figsize=(10, 6))
df_album.boxplot(column="log_track_views", by="album", rot=90)
plt.xlabel("Album")
plt.ylabel("Log Track Views")
plt.title("Log-Transformed Track Views by Album")
plt.suptitle("")  # remove automatic pandas title
plt.tight_layout()
plt.show()

# Certain albums exhibit higher median popularity than others, highlighting substantial differences in track performance across releases.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

features = [
    "word_count",
    "unique_word_ratio",
    "avg_word_length",
    "sentiment_score",
    "first_person_pronoun_ratio",
    "explicit_word_ratio",
]

X = df[features]
y = df["log_track_views"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("R^2 on test set:", round(r2, 3))

coef_df = pd.DataFrame({
    "feature": features,
    "coefficient": model.coef_
}).sort_values("coefficient", key=abs, ascending=False)

print(coef_df)


# A linear regression model was fit to access associations between lyrical features and log-transformed track views.
# The model explains a moderate portion of variance (R^2 ≈ 0.13), indicating that lyrical characteristics alone affect
# popularity differences to a small degree. Coefficient sizes suggest that explicit word usage and
# first-person pronoun frequency show positive associations with popularity, while vocabulary diversity shows a
# negative association. We interpret these effects as correlational rather than causal, and substantial
# unexplained variance likely reflects non-lyrical factors such as marketing, timing, and audience exposure.
# word_count and sentiment_score's coefficient values of ~0 confirm our previous analysis.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train_s, X_test_s, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model_std = LinearRegression()
model_std.fit(X_train_s, y_train)

y_pred_s = model_std.predict(X_test_s)
r2_std = r2_score(y_test, y_pred_s)

coef_std = pd.DataFrame({
    "feature": features,
    "standardized_coefficient": model_std.coef_
}).sort_values("standardized_coefficient", key=abs, ascending=False)

print(coef_std)
