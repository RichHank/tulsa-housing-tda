import kmapper as km
import numpy as np

def build_opportunity_graph(data, features):
    """
    Builds an interactive KeplerMapper graph of the 12D data.
    """
    mapper = km.KeplerMapper(verbose=1)
    
    # Use Affordability Index and Opportunity Score as lenses
    lens = mapper.fit_transform(features, projection=[0, 1])
    
    # Create the simplicial complex
    graph = mapper.map(lens, 
                       features,
                       clusterer=km.cluster.DBSCAN(eps=0.5, min_samples=3),
                       cover=km.Cover(n_cubes=15, perc_overlap=0.2))
    
    return mapper, graph

def generate_mapper_html(mapper, graph, output_file="opportunity_mapper.html"):
    mapper.visualize(graph, path_html=output_file, 
                     title="Topological Opportunity Mapper",
                     custom_tooltips=np.arange(len(graph['nodes'])))
