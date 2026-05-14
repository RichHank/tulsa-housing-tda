import numpy as np

def compute_persistence_silhouette(diagram, resolution=100):
    """
    Computes a simple persistence silhouette from a persistence diagram.
    """
    if len(diagram) == 0:
        return np.zeros(resolution)
        
    t = np.linspace(0, 1.0, resolution)
    silhouette = np.zeros(resolution)
    
    for birth, death in diagram:
        if np.isinf(death):
            death = 1.0
        # Triangle function for each point
        midpoint = (birth + death) / 2
        height = (death - birth) / 2
        
        # Add to silhouette
        mask1 = (t >= birth) & (t <= midpoint)
        mask2 = (t > midpoint) & (t <= death)
        
        silhouette[mask1] += (t[mask1] - birth)
        silhouette[mask2] += (death - t[mask2])
        
    return silhouette

def calculate_buy_signal(B_vector, full_dgm, sub_dgm):
    """
    The decision function:
    S(B) = \int_{0}^{\infty} \Lambda_{sub}(t) - \Lambda_{full}(t) dt
    """
    # Compute silhouettes (Lambda)
    lambda_full = compute_persistence_silhouette(full_dgm)
    lambda_sub = compute_persistence_silhouette(sub_dgm)
    
    # Path integral
    integral = np.trapz(lambda_sub - lambda_full, dx=0.01)
    
    return integral
