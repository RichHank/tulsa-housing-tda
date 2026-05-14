# The Tulsa Topological Affordability Spacetime (TTAS)

Welcome to the most sophisticated open-source project ever created for the Tulsa, Oklahoma housing market. TTAS treats the housing market as a **high-dimensional, non-stationary manifold** and utilizes **multiparameter persistent homology**, **causal inference**, and **spectral geometry** to detect latent phase transitions in the Rent-vs-Buy decision boundary.

## The Tulsa Manifold: Data Construction & Embedding

The property space in Tulsa is modeled as a point cloud in $\mathbb{R}^{12}$, encompassing features like median listing price, rent-to-price ratio, inventory velocity, and opportunity indices.

We employ **Diffusion Maps** and **UMAP** to uncover the intrinsic manifold geometry, along with **Time-Delay Embeddings (Takens' Theorem)** to capture the market's dynamic phase space attractor.

## Multiparameter Persistence Engine

At the core of TTAS is a tri-parameter filtration over:
1. **$\lambda_1$**: Affordability Index
2. **$\lambda_2$**: Spatial Density
3. **$\lambda_3$**: Opportunity Score

We compute **signed barcodes**, **Hilbert functions**, and the **Euler Characteristic Surface** $\chi(\lambda_1, \lambda_2, \lambda_3)$, representing the alternating sum of Betti numbers.

## Rent-vs-Buy Topological Decision Boundary

The "Buy" signal is defined as a Path Integral through the Persistence Landscape:
$$ S(B) = \int_{0}^{\infty} \Lambda_{\text{sub}}(t) - \Lambda_{\text{full}}(t) \, dt $$
Where $\Lambda$ represents the persistence landscapes.

We also train a Gaussian Process Classifier on the Euler Characteristic Surface to detect market regime shifts.

## Causal Topological Inference

Using methods like Wasserstein distance between factual and counterfactual persistence diagrams (Topological ATE), we estimate the causal effects of macroeconomic shocks, such as interest rate hikes, on the market topology.

## Quickstart (Docker)

```bash
cd ttas/docker
docker-compose up --build
```
Navigate to `http://localhost:8050` to explore the cinematic Plotly Dash dashboard.

## Deliverable Structure
Please review the internal `data/`, `topology/`, `decision/`, `visualizations/`, and `dashboard/` directories for the comprehensive mathematical implementation.
