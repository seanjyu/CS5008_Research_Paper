"""
brute_force

This script implements the brute force solution to the travelling salesman
problem. Solution inspired from leetcode solution:
https://leetcode.com/problems/permutations/solutions/2970539/super-simple-brute-force-beats-90-python/?q=brute&orderBy=most_relevant&languageTags=python3
However, heavily edited to fit the travelling salesman problem.
"""
from utilities import graph_class


def brute_force(graph):
    """
    Brute Force algorithm for travelling salesman
    :param graph: graph object from utilities
    :return: permuations of valid tours, minimum weight
    """
    nodes = graph.nodes
    permutations = []
    weights = []

    # find every permutation of nodes in graph using recursive function
    def add_to_permutation(current, remaining, solution, current_weight,
                           solution_weights):
        # if no more remaining add to solution list
        if len(remaining) == 0:
            # if reverse already in list do not append to solution list
            if current[::-1] not in solution:
                current_weight += graph.get_edge_weight(current[0],
                                                        current[-1])
                solution.append(current)
                solution_weights.append(current_weight)
            return

        # add a new node to current list
        for index in range(len(remaining)):
            add_to_permutation(current + [remaining[index]],
                               remaining[:index]
                               + remaining[index + 1:], solution,
                               current_weight + graph.get_edge_weight(
                                   current[-1], remaining[index]),
                               solution_weights)

    # start with first node and add every other node in list
    for index in range(len(nodes)):
        add_to_permutation([nodes[index]], nodes[:index]
                           + nodes[index + 1:], permutations,
                           0, weights)

    return permutations, min(weights)