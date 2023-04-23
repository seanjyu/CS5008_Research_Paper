from utilities import graph_class
from utilities import priority_queue
import networkx as nx

def prims_algorithm(graph):

    # keep set of visited nodes
    visited = []

    # initialize list to store minimum spanning tree
    mst_edges = []

    # create priority queue to store edge weights, priority queue will be
    # ordered by edge weight such that when an element is popped it will
    # have the lowest value.
    queue = priority_queue()

    # start at first node in list
    visited.append(graph.nodes[0])

    # add edges from first node to priority queue
    for index, weight in enumerate(graph.adjacency_matrix[0]):
        if weight != 0:
            queue.insert(weight, graph.nodes[0], graph.nodes[index])

    while not queue.is_empty():
        # pop minimum weight from priority queue.
        # note edge is a tuple in the form of
        # (weight, start node, end node)
        weight, start_node, end_node = queue.dequeue()

        # if end node (3rd element in tuple) is in visited then skip
        if end_node in visited:
            continue

        # visit new node and add minimum edge to the minimum spanning tree
        # note need to reformat tuple
        mst_edges.append((start_node, end_node, weight))
        visited.append(end_node)

        # add edges from new visited node if not visited
        for index, weight in enumerate(graph.adjacency_matrix[graph.nodes
                .index(end_node)]):
            if weight != 0 and graph.nodes[index] not in visited:
                queue.insert(weight, end_node, graph.nodes[index])

    # create new graph object with mst
    mst_obj = graph_class(graph.nodes, mst_edges)
    return mst_obj

def find_odd_degree(mst, graph):
    odd_nodes = []

    for row in range(len(mst.nodes)):
        sum_degrees = 0
        for col in range(len(mst.nodes)):
            if mst.adjacency_matrix[row][col] != 0:
                sum_degrees += 1
        if sum_degrees % 2 == 1:
            odd_nodes.append(mst.nodes[row])

    # create sub graph containing only odd nodes
    number_odd_nodes = len(odd_nodes)
    odd_edges = []

    for i in range(number_odd_nodes):
        for j in range(i + 1, number_odd_nodes):
            odd_edges.append((odd_nodes[i], odd_nodes[j], -1 *
                               graph.get_edge_weight(odd_nodes[i],
                                                     odd_nodes[j])))

    return graph_class(odd_nodes, odd_edges)

def find_minimum_matching_and_eulerian_tour(mst, graph):

    # create networkx graph
    odd_graph_nx = nx.Graph()
    odd_subgraph = find_odd_degree(mst, graph)
    odd_graph_nx.add_weighted_edges_from(odd_subgraph.edges)

    #odd_graph_nx.add_weighted_edges_from(graph.edges)

    matching = list(nx.max_weight_matching(odd_graph_nx,
                                           maxcardinality=True))

    # add back weight information to the odd matching subgraph
    weighted_matching = []
    for pair in matching:
        weighted_matching.append((pair[0], pair[1],
                                  graph.get_edge_weight(pair[0], pair[1])))

    # combine with original graph
    combined_graph = nx.MultiGraph()
    combined_graph.add_weighted_edges_from(mst.edges)

    combined_graph.add_weighted_edges_from(weighted_matching)

    # use networkx method to find eularian tour
    eulerian_tour = list(nx.eulerian_circuit(combined_graph))

    return eulerian_tour

def christofides_serdyukov_algorithm(graph):
    # use prim's algorithm to find mst of graph
    mst = prims_algorithm(graph)

    # find minimum matching and eularian tour
    eulerian_tour = find_minimum_matching_and_eulerian_tour(mst, graph)

    # get unique node path
    tsp_tour_node_order = [eulerian_tour[0][0]]
    for edge in eulerian_tour:
        if edge[1] not in tsp_tour_node_order:
            tsp_tour_node_order.append(edge[1])

    # get edges and calculate total weight
    total_weight = 0
    tsp_tour_edge = []
    for index in range(len(tsp_tour_node_order)):
        node_1 = tsp_tour_node_order[index - 1]
        node_2 = tsp_tour_node_order[index]
        tsp_tour_edge.append((node_1, node_2))
        total_weight += graph.get_edge_weight(node_1, node_2)
    return tsp_tour_edge, total_weight

test1 = graph_class([0, 1, 2, 3, 4], [(0,1,2), (0,3,6), (1, 2, 3), (1,3,8), (1,4,5), (2,4,7), (3,4,9)])
test2 = graph_class(["A", "B", "C"], [("A", "B", 5), ("A", "C", 3)])
test3 = graph_class([1, 2, 3, 4], [(1, 2, 10), (1, 4, 20), (1, 3, 15), (2, 4, 25), (2, 3, 35), (3, 4, 30)])
test4 = graph_class([1, 2, 3, 4, 5], [(1, 2, 12), (1, 4, 19), (1, 3, 10), (1, 5, 8), (2, 3, 3), (2, 4, 7), (2, 5, 2), (3, 4, 6), (3, 5, 20), (4, 5, 4)])
# print(christofides_serdyukov_algorithm(test3))
# print(christofides_serdyukov_algorithm(test4))
# mst_test3 = prims_algorithm(test3)
mst_test4 = prims_algorithm(test4)