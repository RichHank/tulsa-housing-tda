import numpy as np
import pandas as pd
import kmapper as km
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add data folder to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/data")
from fetch_data import synthesize_tulsa_manifold

# 1. Fetch REAL Data
df = synthesize_tulsa_manifold()
features = ['price', 'rent_to_price', 'velocity', 'tax', 'school', 'centrality', 'amenity', 'crime', 'flood', 'walk', 'mobility', 'dti']
data = df[features].values

# Calculate properties
df['opportunity_score'] = (df['school'] / 10.0) + (df['mobility'] / df['mobility'].max())
df['affordability_index'] = df['rent_to_price'] * (1.0 - df['dti'])

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 2. KeplerMapper
mapper = km.KeplerMapper(verbose=1)

# Lens: Opportunity and Affordability
lens = np.column_stack((df['opportunity_score'], df['affordability_index']))

# Map
graph = mapper.map(lens, 
                   data_scaled,
                   clusterer=km.cluster.DBSCAN(eps=1.2, min_samples=3),
                   cover=km.Cover(n_cubes=15, perc_overlap=0.4))

# Custom tooltips with real dates
tooltips = np.array([f"Date: {date.strftime('%Y-%m')} <br> Price Index: {price:.1f} <br> 30Y Mortgage: {mort:.2f}% <br> Unemp: {unemp:.1f}%" 
                     for date, price, mort, unemp in zip(df['Date'], df['price'], df['mortgage'], df['unemployment'])])

mapper.visualize(graph, path_html="tulsa_real_opportunity_mapper.html", 
                 title="Tulsa Topological Opportunity Mapper (REAL DATA)",
                 color_values=df['price'],
                 color_function_name="Tulsa House Price Index (HPI)",
                 custom_tooltips=tooltips)
                 
print("Mapper HTML generated with real data.")
