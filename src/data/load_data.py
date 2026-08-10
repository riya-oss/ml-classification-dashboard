import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded: {path}")
    print(f"Shape : {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def load_excel(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    print(f"Loaded: {path}")
    print(f"Shape : {df.shape[0]} rows x {df.shape[1]} columns")
    return df
