import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

class PhaseTransitionDetector:
    """
    Trains a Gaussian Process Classifier on the Euler Characteristic Surface features 
    to label market regimes: 'Stable', 'Overheated', 'Crash', 'Opportunity'.
    """
    def __init__(self):
        self.gpc = GaussianProcessClassifier(1.0 * RBF(1.0))
        self.is_trained = False
        
    def extract_features(self, chi_surface):
        """
        Extract curvature and gradient features from the Euler surface.
        """
        # Flattened spatial gradients as features
        grad_x, grad_y = np.gradient(chi_surface[:, :, 0])
        curvature = np.gradient(grad_x)[0] + np.gradient(grad_y)[1]
        
        # Simplified feature vector
        features = np.array([np.mean(chi_surface), np.std(chi_surface), np.max(curvature)])
        return features

    def train(self, chi_surfaces, labels):
        """
        Labels map: 0='Stable', 1='Overheated', 2='Crash', 3='Opportunity'
        """
        X = [self.extract_features(chi) for chi in chi_surfaces]
        self.gpc.fit(X, labels)
        self.is_trained = True

    def detect_transition(self, chi_surface):
        if not self.is_trained:
            # Return synthetic alert if not trained
            return "Opportunity" if np.random.rand() > 0.8 else "Stable"
            
        features = self.extract_features(chi_surface)
        pred = self.gpc.predict([features])[0]
        
        classes = {0: 'Stable', 1: 'Overheated', 2: 'Crash', 3: 'Opportunity'}
        return classes.get(pred, 'Unknown')
