import os
import subprocess
import numpy as np

def main():
    print("=========================================================")
    print(" Tulsa Topological Affordability Spacetime (TTAS) Engine")
    print("=========================================================")
    
    print("\n[1/5] Synthesizing 12D Tulsa Manifold Data...")
    from data.fetch_data import synthesize_tulsa_manifold
    from data.preprocess import preprocess_manifold
    df = synthesize_tulsa_manifold()
    df = preprocess_manifold(df)
    print(f"      -> Generated {len(df)} points.")
    
    print("\n[2/5] Computing Manifold Embeddings...")
    from data.embeddings import EmbeddingEngine
    ee = EmbeddingEngine(df)
    umap_emb = ee.compute_umap()
    print("      -> UMAP 3D Embedding generated.")
    
    print("\n[3/5] Constructing Multiparameter Filtrations...")
    from topology.filtrations import TriParameterFiltration
    fil = TriParameterFiltration(umap_emb, df['affordability_index'], np.zeros(len(df)), df['opportunity_score'])
    fil_data = fil.construct_filtration()
    print(f"      -> Filtration computed via {fil_data.get('backend')}.")
    
    print("\n[4/5] Evaluating Path Integral Decision Boundary...")
    from decision.path_integral import calculate_buy_signal
    S_B = calculate_buy_signal(None, [[0.1, 0.5], [0.2, 0.8]], [[0.15, 0.4]])
    print(f"      -> Simulated Buy Signal S(B) = {S_B:.4f}")
    
    print("\n[5/5] Launching Cinematic Interactive Dashboard...")
    print("      -> Dashboard available at http://localhost:8050")
    
    # Run the dash app
    subprocess.run(["python", "dashboard/app.py"])

if __name__ == "__main__":
    main()
