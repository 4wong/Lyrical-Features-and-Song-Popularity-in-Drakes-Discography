## TL;DR
This project analyses how basic lyrical features relate to song popularity across Drake's discography using lyrics sourced from Genius. 
A linear regression model finds that lyrical features explain a modest share of popularity (R² ≈ 0.13), with simpler and more repetitive lyrics showing stronger associations with higher popularity than sentiment or song length. 
Overall, the results suggest that lyrical structure (simplicity, repetition, stylistic choices) matters more than lyrical tone (sentiment of words), while most variation in popularity is driven by non-lyrical factors such as marketing, release context, and platform dynamics.

## Project Overview
This project explores how basic lyrical characteristics such as length, vocabulary diversity, and sentiment relate to song popularity across Drake’s discography using lyrics sourced from Genius.

---

## Research Question
**How do basic lyrical characteristics (e.g. word count, vocabulary diversity, and sentiment) relate to song popularity across Drake’s discography?**

---

## Data Source
Lyrics and metadata were sourced from a Kaggle dataset compiled from Genius in **2021**. The dataset represents a static snapshot and may not reflect later lyric revisions or updated popularity metrics, particularly for unreleased tracks.

---

## Methods
Lyrics from each song were cleaned and tokenised before being turned into features such as word count, unique words, sentiment (VADER), etc.  
Track views were log-transformed in order to address the right skew present.  
Exploratory data analysis and linear regression were both used to examine associations between these created features and popularity.

---

## Key Findings

My regression model only explains 13% of the variance in popularity, with an R-squared value of roughly 0.13.  
I expected this to be the case, as I believe a song’s popularity has many other components that make up the majority of the variance, such as marketing for hype, release timing, platform algorithms, etc.  
Despite this limitation, our results suggest that lyrical structure is more informative than lyrical tone in explaining the differences in popularity within songs.

Our standardised coefficients show that simpler and more repetitive lyrics are associated with higher popularity in a song. Songs with more complex or diverse vocabularies tend to perform slightly worse. This makes a lot of sense, as an artist that bridges multiple genres, Drake appeals to a vast variety of audiences rather than being just a “rapper”. His more lyrical, introspective tracks which showcase his artistic expression targeted toward his die-hard fans will appeal to fewer people, while his highest-performing tracks that are catchy and repetitive such as *Hotline Bling*, *One Dance*, and *God’s Plan* will skyrocket in the pop music scene.

These results reflect how mass appeal in mainstream music favours memorability and catchiness with minimal complexity over technical depth.


