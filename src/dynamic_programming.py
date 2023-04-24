"""
dynmic_programming

Implementation of dynamic programming algorithm to solve for the travelling
salesman problem. The
"""

from utilities import graph_class
import sys

class dynamic_programming:
    def __init__(self, graph):
        """
        Initialize object
        :param graph: graph class from utilities
        """
        self.adjacency_matrix = graph.adjacency_matrix
        self.min_tour_cost = sys.maxsize
        self.run_solver = False
        self.start_node = 0
        self.length = len(graph.adjacency_matrix)
        self.subsets = []
        self.min_tour =[]
        self.solve()


    def get_min_tour_cost(self):
        """
        Getter method for min_tour_cost
        :return: int representing min tour cost
        """
        return self.min_tour_cost

    def solve(self):
        """
        method to solve travelling salesman problem using dynamic
        programming
        :return: void
        """
        # if solver already run then just return
        if self.run_solver:
            return

        # define final state
        end_state = (1 << self.length) - 1

        # create memoization for dynamic programming
        memo = [[0 for i in range((1 << self.length))]
                for i in range(self.length)]

        # copy first row of adjacency matrix values matrix
        for i in range(1, self.length):
            memo[i][(1 << 0 | 1 << i)] = self.adjacency_matrix[0][i]

        for r in range(3, self.length + 1):
            self.combinations(r, self.length)
            # for subset in self.combinations(r, self.length):
            for subset in self.subsets:
                # if start node not in subset then skip
                if self.bit_not_in(0, subset):
                    continue
                # parse through next elements
                for next in range(1, self.length):
                    if self.bit_not_in(next, subset):
                        continue

                    subset_without_next = subset ^ (1 << next)
                    min_dist = sys.maxsize

                    for end in range(1, self.length):
                        if end == next or self.bit_not_in(end, subset):
                            continue
                        # calculate distance
                        new_distance = memo[end][subset_without_next] \
                                       + self.adjacency_matrix[end][next]

                        if new_distance < min_dist:
                            min_dist = new_distance

                    # add to memoization
                    memo[next][subset] = min_dist

        # calculate minimum cost
        for i in range(1, self.length):
            tour_cost = memo[i][end_state] + self.adjacency_matrix[i][0]
            if tour_cost < self.min_tour_cost:
                self.min_tour_cost = tour_cost

        # Using memo find optimal tour
        self.min_tour.append(0)
        prev_node = 0
        state = end_state
        for i in range(self.length - 1, 0, -1):
            optimal_node = -1
            optimal_dist = sys.maxsize
            for j in range(self.length):
                if self.bit_not_in(j, state):
                    continue
                # calculate distance
                if optimal_node == -1:
                    optimal_node = j
                prev_distance = memo[optimal_node][state] \
                                + self.adjacency_matrix[optimal_node][
                                    prev_node]
                new_distance = memo[j][state] \
                               + self.adjacency_matrix[j][prev_node]
                if new_distance < prev_distance:
                    optimal_node = j

            self.min_tour.append(optimal_node)
            state = state ^ (1 << optimal_node)
            prev_node = optimal_node

        # append first node at end since need to start and end on same node
        self.min_tour.append(0)
        self.run_solver = True


    def combinations(self, r, n):
        """
        method to call recursive function and output combinations
        :param r: current number of nodes
        :param n: total number of nodes
        :return: list of subsets represtented by binary number
        """
        self.subsets = []
        self.recursive_combinations(0, 0, r, n, self.subsets)
        return self.subsets

    def recursive_combinations(self, set, at, r, n, subsets):
        """
        Recursive function to find combinations
        :param set: current set
        :param at: current node
        :param r: number of nodes
        :param n: total number of nodes
        :param subsets: list of subsets
        :return: void
        """
        # if no more elements to add return
        if n - at < r:
            return

        # r elements selected so valid subset found, add to subset list
        if r == 0:
            self.subsets.append(set)
        #
        else:
            for i in range(at, n):
                set ^= (1 << i)

                self.recursive_combinations(set, i + 1, r - 1, n, subsets)

                set ^= (1 << i)

    def bit_not_in(self, node, subset):
        """
        Check if node is in set that is represented by binary number
        :param node:
        :param subset:
        :return:
        """
        return ((1 << node) & subset) == 0
