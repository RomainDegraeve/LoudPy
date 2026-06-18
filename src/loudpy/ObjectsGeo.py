import numpy as np 
import inspect
import json

class Nodes:
    def __init__(self, node_tags: np.ndarray, node_coords: np.ndarray):
        self.node_tags = node_tags
        self.node_coords = node_coords

class L3_Edges(Nodes):
    def __init__(self, node_tags, node_coords, edges: np.ndarray, name: str):
        super().__init__(node_tags, node_coords)
        self.edges = edges
        self.name = name

class TR6_Surfs(Nodes):
    def __init__(self, node_tags, node_coords, tri: np.ndarray, name: str):
        super().__init__(node_tags, node_coords)
        self.tri = tri
        self.name = name

    @classmethod
    def get_params(cls, mat_name: str, mat_path):

        with open(mat_path, "r") as f:
            data = json.load(f)[mat_name]

        # L'inspection se base sur 'cls', donc sur la classe finale appelée
        sig = inspect.signature(cls).parameters
        
        return {k: v for k, v in data.items() if k in sig}