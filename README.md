# Uber pickups in New York, April 2014

A Streamlit page for 564,516 Uber pickups in New York during April 2014. It shows when people called a car and where they were standing.

The file `uber-raw-data-apr14.csv` must sit next to `app.py`.

Use **Python 3.11 or newer**. Streamlit Community Cloud currently runs Python 3.14, and these versions install there as well as on 3.11.

## Run

From this folder, in a new environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

With `make` available and the environment already active:

```text
make install
make run
```

The app opens at http://localhost:8501.

## Dependencies

`requirements.txt` pins the libraries this project imports: Streamlit, pandas, NumPy, Matplotlib, Seaborn, missingno, and pydeck. pyarrow, plotly, altair, and requests are not used.
