import pandas as pd


def check_shape(df: pd.DataFrame) -> None:
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")


def check_nulls(df: pd.DataFrame) -> None:
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]

    if cols_with_nulls.empty:
        print("Nulls: none found")
    else:
        print("Nulls found:")
        for col, count in cols_with_nulls.items():
            pct = count / len(df) * 100
            print(f"  {col}: {count} ({pct:.1f}%)")


def check_target(df: pd.DataFrame, target_col: str) -> None:
    if target_col not in df.columns:
        print(f"Target column '{target_col}' NOT found in dataset")
        return

    print(f"Target column '{target_col}' found")
    print(f"  dtype       : {df[target_col].dtype}")
    print(f"  unique vals : {df[target_col].nunique()}")
    print(f"  distribution:\n{df[target_col].value_counts(normalize=True).mul(100).round(1).to_string()}")


def run_all(df: pd.DataFrame, target_col: str) -> None:
    print("=== Shape ===")
    check_shape(df)
    print("\n=== Nulls ===")
    check_nulls(df)
    print("\n=== Target ===")
    check_target(df, target_col)
