import numpy as np
from pydiffmap import diffusion_map
import umap
from sklearn.preprocessing import StandardScaler

class EmbeddingEngine:
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()
        
    def get_features(self):
        features = self.data[['price', 'rent_to_price', 'velocity', 'tax', 'school', 
                              'centrality', 'amenity', 'crime', 'flood', 'walk', 
                              'mobility', 'dti']].values
        return self.scaler.fit_transform(features)

    def compute_diffusion_map(self):
        """
        Reveal the intrinsic manifold of affordability similarity.
        """
        X = self.get_features()
        mydmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs=3, epsilon=1.0, alpha=0.5)
        dmap_embedding = mydmap.fit_transform(X)
        return dmap_embedding

    def compute_umap(self, supervised=False, labels=None):
        """
        Generate a 3D embedding for interactive exploration.
        """
        X = self.get_features()
        reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1)
        if supervised and labels is not None:
            embedding = reducer.fit_transform(X, y=labels)
        else:
            embedding = reducer.fit_transform(X)
        return embedding

    def takens_embedding(self, series, delay=1, dimension=3):
        """
        Time-Delay Embedding (Takens) for price trajectories.
        """
        n = len(series)
        embedded = np.zeros((n - (dimension - 1) * delay, dimension))
        for i in range(dimension):
            embedded[:, i] = series[i * delay : i * delay + embedded.shape[0]]
        return embedded
