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
