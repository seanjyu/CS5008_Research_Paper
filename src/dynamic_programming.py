from utilities import graph_class
import sys

class dynamic_programming:
    def __init__(self, graph):
        self.adjacency_matrix = graph.adjacency_matrix
        self.min_tour_cost = sys.maxsize
        self.run_solver = False
        self.start_node = 0
        self.length = len(graph.adjacency_matrix)
        self.subsets = []
        self.min_tour =[]
        self.solve()


    def get_min_tour_cost(self):
        return self.min_tour_cost
    
    def solve(self):
        if self.run_solver == True:
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
        for i in range(self.length):
            optimal_node = -1
            optimal_dist = sys.maxsize
            for j in range(self.length):
                if self.bit_not_in(j, state):
                    continue
                # calculate distance
                new_distance = memo[j][state] \
                               + self.adjacency_matrix[j][prev_node]

                if new_distance < optimal_dist:
                    optimal_node = j
                    optimal_dist = new_distance

            self.min_tour.append(optimal_node)
            state = state ^ (1 << optimal_node)
            prev_node = optimal_node

        self.min_tour.append(0)
        self.run_solver = True

    def combinations(self, r, n):
        self.subsets = []
        self.recursive_combinations(0, 0, r, n, self.subsets)
        return self.subsets

    def recursive_combinations(self, set, at, r, n, subsets):

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

                set ^= (1 << i);

    def bit_not_in(self, node, subset):
        return ((1 << node) & subset) == 0;

test3 = graph_class([1, 2, 3, 4], [(1, 2, 10), (1, 4, 20), (1, 3, 15), (2, 4, 25), (2, 3, 35), (3, 4, 30)])
test4 = graph_class([1, 2, 3, 4, 5], [(1, 2, 12), (1, 4, 19), (1, 3, 10), (1, 5, 8), (2, 3, 3), (2, 4, 7), (2, 5, 2), (3, 4, 6), (3, 5, 20), (4, 5, 4)])

a = dynamic_programming(test3)
b = dynamic_programming(test4)
a.solve()
b.solve()
print(a.min_tour_cost, b.min_tour_cost)