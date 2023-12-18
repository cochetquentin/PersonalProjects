import networkx as nx
import numpy as np

class Graph():
    def __init__(self, game_matrix:np.ndarray) -> None:
        self.game_matrix = game_matrix
        self.adj_matrix = self.array_to_adj_matrix()

        self.graph = nx.from_numpy_array(self.adj_matrix)
        self.rename_nodes()
        
    def array_to_adj_matrix(self):
        shape = self.game_matrix.shape
        N = shape[0] * shape[1]

        # Create an adjacency matrix filled with zeros
        adjacency_matrix = np.zeros((N, N), dtype=int)

        # Helper function to convert 2D indices to a flat index
        def index(i, j):
            return i * shape[1] + j

        # Iterate through the array to fill the adjacency matrix
        for i in range(shape[0]):
            for j in range(shape[1]):
                if self.game_matrix[i, j] == 1:  # Obstacle, skip
                    continue

                # Check left neighbor
                if j > 0 and self.game_matrix[i, j - 1] != 1:
                    adjacency_matrix[index(i, j), index(i, j - 1)] = 1

                # Check right neighbor
                if j < shape[1] - 1 and self.game_matrix[i, j + 1] != 1:
                    adjacency_matrix[index(i, j), index(i, j + 1)] = 1

                # Check top neighbor
                if i > 0 and self.game_matrix[i - 1, j] != 1:
                    adjacency_matrix[index(i, j), index(i - 1, j)] = 1

                # Check bottom neighbor
                if i < shape[0] - 1 and self.game_matrix[i + 1, j] != 1:
                    adjacency_matrix[index(i, j), index(i + 1, j)] = 1

        return adjacency_matrix
    
    def rename_nodes(self):
        mapping = {i*self.game_matrix.shape[1] + j: (i, j) for i in range(self.game_matrix.shape[0]) for j in range(self.game_matrix.shape[1])}
        self.graph = nx.relabel_nodes(self.graph, mapping)
    
    def get_shortest_path(self, source_coord:tuple, destination_coord:tuple):
        try:
            return nx.shortest_path(self.graph, source_coord, destination_coord)
        except nx.NetworkXNoPath:
            return []
    
    @staticmethod
    def get_shortest_path_static(graph, source_coord:tuple, destination_coord:tuple):
        try:
            return nx.shortest_path(graph, source_coord, destination_coord)
        except nx.NetworkXNoPath:
            return []
        
    def plot(self):
        nx.draw(self.graph, with_labels=True)

    @staticmethod
    def distance_between(source, target):
        return abs(source[0] - target[0]) + abs(source[1] - target[1])
    
    @staticmethod
    def most_far_away_node(graph, source):
        nodes = list(graph.nodes)
        distances = [Graph.distance_between(source, node) for node in nodes]
        return nodes[np.argmax(distances)]
    
    def get_longest_path(self, source):
        nodes = nx.descendants(self.graph, source)
        nodes.add(source)
        g = self.graph.subgraph(nodes)
        print("RECHERCHE DU SOMMET LE PLUS LOIN")
        mfan = Graph.most_far_away_node(g, source)
        if len(nodes) > 100:
            return nx.shortest_path(g, source, mfan)
        else:
            print("RECHERCHE DE TOUS LES CHEMINS")
            paths = nx.all_simple_paths(g, source, mfan)
            for path in paths:
                return path