import tsfeatures

def compute_tsfeatures(df, freq):
    assert df.columns in ["unique_id", "ds", "y"], '`df` should have ["unique_id", "ds", "y"] as columns'
    return tsfeatures(df, freq=freq).set_index("unique_id")