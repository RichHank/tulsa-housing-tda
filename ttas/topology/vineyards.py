import numpy as np
import persim
from ripser import ripser

class PersistenceVineyard:
    """
    Constructs Persistence Vineyards over the entire 7-year span, 
    tracking the birth/death of H0 and H1 features.
    """
    def __init__(self, time_series_point_clouds):
        self.point_clouds = time_series_point_clouds # List of point clouds per month
        self.diagrams = []

    def compute_vineyard(self):
        for pc in self.point_clouds:
            dgm = ripser(pc, maxdim=1)['dgms']
            self.diagrams.append(dgm)
        return self.diagrams

    def compute_sliding_window_bottleneck(self, window_size=12):
        """
        Use a sliding window (12-month) with 1-month stride. 
        For each window, compute the Bottleneck distance to a stable baseline.
        """
        baseline_dgm = self.diagrams[0][1] # H1 of first month
        distances = []
        
        for i in range(len(self.diagrams) - window_size + 1):
            # Aggregate or average diagram for window
            # Simplified: just take the first diagram in the window
            window_dgm = self.diagrams[i][1] 
            
            # persim bottleneck needs matched dimensions
            if len(baseline_dgm) > 0 and len(window_dgm) > 0:
                dist = persim.bottleneck(baseline_dgm, window_dgm)
            else:
                dist = 0.0
            distances.append(dist)
            
        return distances

    def topological_change_point_detection(self, distances):
        """
        Detect regime shifts using Bayesian blocks or thresholding.
        """
        threshold = np.mean(distances) + 2 * np.std(distances)
        change_points = [i for i, d in enumerate(distances) if d > threshold]
        return change_points
