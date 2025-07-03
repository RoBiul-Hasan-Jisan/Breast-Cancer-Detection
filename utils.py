import pandas as pd

REQUIRED_COLUMNS = [
    'id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',
    'symmetry_mean', 'fractal_dimension_mean'
]

def load_and_validate_csv(file):
    try:
        df = pd.read_csv(file)
        if not all(col in df.columns for col in REQUIRED_COLUMNS):
            return None, f"Missing columns: {set(REQUIRED_COLUMNS) - set(df.columns)}"
        return df, None
    except Exception as e:
        return None, str(e)
