from tsfeatures import tsfeatures
import numpy as np

def compute_tsfeatures(df, freq):
    assert np.all([c in df.columns for c in ["unique_id", "ds", "y"]]), '`df` should have ["unique_id", "ds", "y"] as columns'
    return tsfeatures(df, freq=freq).set_index("unique_id")