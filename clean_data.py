# make_bechdel_clean.py
# Builds bechdel_clean.csv for the dashboard.
# Setup:  pip install pandas numpy requests
#         set your TMDB key first:  export TMDB_API_KEY="..."   (Windows: setx TMDB_API_KEY "...")
# Run:    python make_bechdel_clean.py

import os
import html
import json
import time
import numpy as np
import pandas as pd
import requests

API_KEY = os.environ["TMDB_API_KEY"] 
URL = "https://raw.githubusercontent.com/fivethirtyeight/data/master/bechdel/movies.csv"
MIN_GENRE_FILMS = 20                   # smaller genres get grouped as "Other"
log = []

# ---------- 1. load the data ----------
df = pd.read_csv(URL)
df = df.rename(columns={"budget_2013$": "budget_2013", "intgross_2013$": "intgross_2013"})
df["title"] = df["title"].apply(html.unescape)      # titles have things like &amp; in them
log.append(f"Loaded {len(df)} films from FiveThirtyEight.")

# ---------- 2. fix the IMDb ids (e.g. tt420238 should be tt0420238) ----------
def fix_id(x):
    return "tt" + x[2:].lstrip("0").zfill(7)

df["imdb_id"] = df["imdb"].apply(fix_id)
n_fixed = (df["imdb"] != df["imdb_id"]).sum()
log.append(f"Re-padded {n_fixed} IMDb ids.")

# ---------- 4. get each film's main genre from TMDB ----------
def tmdb_get(url, params):
    r = requests.get(url, params=params)
    while r.status_code == 429:          # too many requests: wait and try again
        time.sleep(2)
        r = requests.get(url, params=params)
    return r.json()

genre_list = tmdb_get("https://api.themoviedb.org/3/genre/movie/list", {"api_key": API_KEY})["genres"]
genre_name = {g["id"]: g["name"] for g in genre_list}

cache_file = "tmdb_cache.json"           # saves results so we only ask TMDB once per film
cache = json.load(open(cache_file)) if os.path.exists(cache_file) else {}

for i, imdb_id in enumerate(df["imdb_id"]):
    if imdb_id in cache:
        continue
    data = tmdb_get(f"https://api.themoviedb.org/3/find/{imdb_id}",
                    {"api_key": API_KEY, "external_source": "imdb_id"})
    results = data.get("movie_results", [])
    cache[imdb_id] = results[0]["genre_ids"] if results else []
    if i % 200 == 0:
        print(i, "of", len(df))
        json.dump(cache, open(cache_file, "w"))
    time.sleep(0.05)
json.dump(cache, open(cache_file, "w"))

def first_genre(imdb_id):
    ids = cache[imdb_id]
    return genre_name[ids[0]] if ids else "Unknown"

df["primary_genre"] = df["imdb_id"].apply(first_genre)
log.append("Primary genre = first genre TMDB lists for the film ('Unknown' if none).")

# ---------- 5. new columns the dashboard needs ----------
# pass = "ok" only; dubious counts as NOT passing
df["pass_group"] = np.where(df["clean_test"] == "ok", "Pass", "Not pass")
log.append("Pass = clean_test 'ok'. Dubious counts as not passing.")

# ROI = gross per $1 of budget (both already in 2013 dollars)
df["roi"] = df["intgross_2013"] / df["budget_2013"]
log.append(f"{df['roi'].isna().sum()} films have no gross, so no ROI (left out of ROI only).")

# 5-year periods (the last one is only 4 years)
edges = [1969, 1974, 1979, 1984, 1989, 1994, 1999, 2004, 2009, 2013]
labels = ["1970–74", "1975–79", "1980–84", "1985–89", "1990–94", "1995–99", "2000–04", "2005–09", "2010–13"]
df["period"] = pd.cut(df["year"], bins=edges, labels=labels).astype(str)

# budget tiers: split into three equal-sized groups
df["budget_tier"] = pd.qcut(df["budget_2013"], 3, labels=["Low", "Mid", "High"]).astype(str)

# group small genres (and Unknown) into "Other"
counts = df["primary_genre"].value_counts()
big_genres = counts[counts >= MIN_GENRE_FILMS].index
df["genre_group"] = np.where(df["primary_genre"].isin(big_genres), df["primary_genre"], "Other")
log.append(f"Genres with fewer than {MIN_GENRE_FILMS} films (and Unknown) grouped as 'Other'.")

# ---------- 6. save ----------
columns = ["imdb_id", "title", "year", "period", "clean_test", "pass_group",
           "budget_2013", "intgross_2013", "roi", "budget_tier", "primary_genre", "genre_group"]
clean = df[columns]
clean.to_csv("bechdel_clean.csv", index=False)

with open("cleaning_log.txt", "w", encoding="utf-8") as f:
    f.write("\n".join("- " + line for line in log))

print("Saved bechdel_clean.csv:", clean.shape)
print(clean["genre_group"].value_counts())
print("\n".join(log))