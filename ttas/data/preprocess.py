import pandas as pd
from sklearn.preprocessing import RobustScaler

def preprocess_manifold(df):
    """
    Clean and normalize the synthesized data for TDA pipeline.
    """
    scaler = RobustScaler()
    cols_to_scale = ['price', 'velocity', 'crime', 'flood', 'walk', 'mobility']
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    # Calculate Opportunity Score (lambda_3 parameter)
    df['opportunity_score'] = (df['school'] / 10.0) + (df['mobility'] / df['mobility'].max())
    
    # Calculate Affordability Index (lambda_1 parameter)
    df['affordability_index'] = df['rent_to_price'] * (1.0 - df['dti'])
    
    return df
