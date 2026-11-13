## app.py  --  Bechdel dashboard (Plotly Dash)

Needs bechdel_clean.csv in the same folder (made by make_bechdel_clean.py).

Setup:  pip install dash pandas numpy plotly

Run:    python app.py      then open  http://127.0.0.1:8050

### ONLY RUN APP.PY FOR THE DASHBOARD

## clean_data.py

Builds bechdel_clean.csv for the dashboard.

Setup:  pip install pandas numpy requests
        
        set your TMDB key first:  export TMDB_API_KEY="..."   (Windows: setx TMDB_API_KEY "...")

Run:    python clean_data.py
