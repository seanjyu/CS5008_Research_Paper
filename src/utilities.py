"""
utilities

File that contains utilities for algorithms which consist of:
graph_class: Class that represents graph data structure
priority_queue: priority queue implemented using built-in heapq class
"""

import heapq

class graph_class:
    def __init__(self, nodes, edges):
        """
        Initialize graph class object.
        :param nodes: List of strings that represent nodes
        :param edges: List of tuples in the format of
        (node name (string), node name (string), weight (integer))
        """
        self.nodes = nodes
        self.edges = edges
        self.adjacency_matrix = self.create_adjacency_matrix(nodes, edges)


    def create_adjacency_matrix(self, nodes, edges):
        """
        Method to create adjacency matrix
        :param nodes:
        :param edges:
        :return: 2D list representing adjacency matrix
        """
        # create empty 2D adjacency matrix
        # Note need to use for loop to create new list objects
        adjacency_matrix = [[0 for i in range(len(nodes))]
                            for i in range(len(nodes))]

        # Loop through the list of edges and assign values to 2D adjacency
        # matrix then update neighbor dictionary.
        for edge in edges:
            index_1 = nodes.index(edge[0])
            index_2 = nodes.index(edge[1])
            adjacency_matrix[index_1][index_2] = edge[2]
            adjacency_matrix[index_2][index_1] = edge[2]
        return adjacency_matrix

    def get_edge_weight(self, node_1, node_2):
        """
        Getter method for edge weight given two nodes
        :param node_1:
        :param node_2:
        :return:
        """
        return self.adjacency_matrix[self.nodes
                                   .index(node_1)][self.nodes
                                   .index(node_2)]

class priority_queue:
    def __init__(self):
        self.queue = []

    def is_empty(self):
        """
        Method to check if queue is empty
        :return: void
        """
        return len(self.queue) == 0

    def insert(self, weight, node_1, node_2):
        """
        Method to insert edges in queue.
        :param weight: weight of edge
        :param node_1: start node
        :param node_2: end node
        :return: void
        """
        heapq.heappush(self.queue, (weight, node_1, node_2))

    def dequeue(self):
        """
        Method to dequeue element with highets priority, in this case the
        lowest weight.
        :return:
        """
        return self.queue.pop(0)
