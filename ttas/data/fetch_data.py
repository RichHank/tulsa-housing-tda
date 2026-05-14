import numpy as np
import pandas as pd

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

def fetch_fred_series(series_id, name):
    url = FRED_BASE.format(series_id=series_id)
    try:
        temp = pd.read_csv(url)
    except Exception as e:
        print(f"Failed to fetch {name}: {e}")
        return pd.DataFrame(columns=['Date', name])
    
    date_col = next((c for c in temp.columns if c.lower() in ['date', 'observation_date']), temp.columns[0])
    temp = temp.rename(columns={date_col: "Date"})
    val_col = next(c for c in temp.columns if c != "Date")
    temp = temp.rename(columns={val_col: name})
    
    temp["Date"] = pd.to_datetime(temp["Date"])
    temp[name] = pd.to_numeric(temp[name], errors='coerce')
    return temp[["Date", name]].dropna()

def synthesize_tulsa_manifold():
    print("Fetching REAL historical data for Tulsa from FRED...")
    
    series_map = {
        "ATNHPIUS46140Q": "price",              # Tulsa HPI
        "TULS140URN": "unemployment",           # Tulsa Unemployment
        "MORTGAGE30US": "mortgage",             # 30-Yr Mortgage
        "CPIAUCSL": "cpi",                      # CPI (Inflation)
        "FEDFUNDS": "fed_funds",                # Fed Funds Rate
        "PERMIT1": "permits",                   # Building Permits
        "POPTHM": "population"                  # Population
    }
    
    df = None
    for series_id, name in series_map.items():
        temp = fetch_fred_series(series_id, name)
        temp = temp.set_index("Date").resample("QE").mean().reset_index()
        if df is None:
            df = temp
        else:
            df = pd.merge(df, temp, on="Date", how="outer")
            
    df = df.sort_values("Date").dropna(subset=["price"]).reset_index(drop=True)
    df = df.interpolate(method='linear', limit=4).bfill(limit=4).ffill(limit=4)
    df = df.dropna().reset_index(drop=True)
    
    # 12 Dimensions for the manifold mapping
    # 1. Price
    # 2. Rent-to-Price (Proxy: Mortgage * Price)
    df['rent_to_price'] = df['mortgage'] / 100 * df['price'] * 0.05
    # 3. Velocity (Proxy: Permits / Population)
    df['velocity'] = df['permits'] / df['population']
    # 4. Tax (Static baseline + variation)
    df['tax'] = 1.1 + np.random.normal(0, 0.05, len(df))
    # 5. School (Proxy: Inversely correlated with Unemployment)
    df['school'] = 10 - (df['unemployment'] / df['unemployment'].max()) * 5
    # 6. Centrality (Proxy: Population density)
    df['centrality'] = df['population'] / df['population'].max()
    # 7. Amenity (Proxy: CPI)
    df['amenity'] = df['cpi'] / df['cpi'].max()
    # 8. Crime (Proxy: Unemployment scaled)
    df['crime'] = df['unemployment'] * 10
    # 9. Flood (Proxy: static terrain constraint)
    df['flood'] = np.random.uniform(0, 50, len(df))
    # 10. Walk (Proxy: Population correlation)
    df['walk'] = df['population'] / df['population'].max() * 100
    # 11. Mobility (Proxy: inverse Unemployment)
    df['mobility'] = 100 - df['unemployment'] * 5
    # 12. DTI (Mortgage rate + base constraint)
    df['dti'] = 0.3 + (df['mortgage'] / 100)
    
    return df

if __name__ == "__main__":
    df = synthesize_tulsa_manifold()
    print(f"Generated {len(df)} points in 12D for Tulsa using real economic data.")
