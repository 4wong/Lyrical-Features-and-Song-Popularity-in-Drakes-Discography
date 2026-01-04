Project Overview:
This project explores how basic lyrical characteristics such as length, voacbulary diversity, and sentiment relate to song popularity across Drake’s discography using lyrics sourced from Genius.

Research Question:
How do basic lyrical characteristics (e.g word count, vocabulary diversity and sentiment) relate to song popularity across Drake’s discography?

Data Source:
Lyrics and metadata were sourced from a Kaggle dataset compiled from Genius in  2021. The dataset represents a static snapshot and may not reflect later lyric revisions or updated popularity metrics, particularly for unreleased tracks.

Methods:
Lyrics from each songs were cleaned and tokenised before turned into features such as word count, unique words, sentiment (VADER), etc.
Track views were log-transformed in order to address the right-skew present.
Exploratory data analysis and linear regression were both used to examine associations between these created features and popularity.

Key Findings:

My regression model only explains 13% of the variance in popularity with an R squared value of roughly 0.13.
I expected this to be the case as I believe a song's popularity has many other components that make up the majority of the variance, such as marketing for hype, release timing, platform algorithms, etc. 
Despite this limitation, our results suggest that lyrical structure is more informative than lyrical tone in explaining the differences in popularity within songs. 

Our standardised coefficients show that more simple and repetitive lyrics are associated with a higher popularity in the song. Song's with more complex or diverse vocabularies tend to perform slightly worse. This makes a lot of sense, as an artist that bridges multiple genres, Drake appeals to a vast variety of audiences rather than being just a "rapper". His more lyrical, introspective tracks which showcase his artistic expression targetted toward his die-hard fans will appeal to less people, while his highest performing tracks that are catchy and repetitive such as Hotline bling, One dance and God's plan will skyrocket in the pop music scene. 

These results reflect how mass appeal in mainstream music favours memorability and catchiness with not much complexity over technical depth.
