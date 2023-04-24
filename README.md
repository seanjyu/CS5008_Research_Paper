# CS5008_Research_Paper
Research Paper on Algorithms for the Travelling Salesman Problem 




The travelling salesman problem can be stated as follows:

*"A traveling salesman wants to visit each of $n$ cities exactly once and return to his starting point. In which order should he visit these cities to travel the minimum total distance?"* <sup>1</sup>

On the surface the problem sounds very simple. However, in reality finding the optimal solution proves to be very difficult, especially as the number of cities increases. 

This problem can be defined mathematically using graphs. With the cities represented by vertexes and the distances between each city represented by weighted vertices. This graph can be considered to be fully connected, meaning that each vertex is connected to each other by an edge. 

Another quality of this graph is that weight of the vertices represents the shortest distance between those two vertexes. This is to say that there cannot exist a path with a shorter distance where an intermediate vertex is visited. This quality is also known as the triangle inequality. Mathematically this quality can be represented as $c(u,w)\leq c(u,v) + c(v,w)$ where $u,v,w$ are vertexes and $c(x,y)$ is a function that returns the weight of the edge between vertex $x$ and vertex $y$. <sup>2</sup>

The condition that the salesman must return to the starting point can be defined mathematically as a tour which is a walk over a graph that has no repeated edges and starts and ends on the same vertex.

Note there are variations to this problem where certain constraints are relaxed, making the problem simpler and easier to solve. However, for the purposes of this paper the above conditions were applied.

A more rigorous mathematical formulation in the form of an integer linear program can be found in the textbook Nonlinear Programming by Dimitri Bertsekas.<sup>3</sup> 

## <ins>History and Importance of the travelling salesman problem</ins>

One of the earliest mentions of this problem could be found in a handbook for travelling through Germany and Switzerland<sup>4</sup>. However, the first mathematical formulation of the problem can be traced to the 19th century by William Rowan Hamilton and Thomas Kirkman. A more general formulation would be studied by Karl Menger in the 1930s in Vienna and Harvard.<sup>5</sup>

This problem is important in the theoretical computer science community since it is one of the problems that is proven to be NP-hard. This means that the optimal solution of this problem can be found in polynomial time but only if there exists an NP-Complete solution with polynomial time that the original problem can reduce to. In the case of the travelling salesman problem, the problem can be reduced to finding the optimal Hamiltonian cycle which is in NP Complete.<sup>6</sup>

As for the applications of this problem, the obvious application of this problem would be in optimization the path for a delivery person to take while delivering packages. However these algorithms have also been used to solve optimal placement of components in printed circuit boards such that the wiring between components can be optimized. <sup>7</sup>


## Implementation and Analysis of Algorithms

<ins>Algorithms Used</ins>

In this report three algorithms were implemented. Namely, the brute force algorithm, the dynamic programming and the Christofides–Serdyukov algorithm. All three algorithms were implemented using python. Note the brute force and dynamic programming algorithms both find the optimal solution since they search the whole solution space. The Christofides-Serdyukov algorithm only provides an estimate. However, because of the accuracy trade-off the Christofides-Serdyukov algorithm is typically many times faster, depending on the implementations of the sub algorithms. This tradeoff is shown very clearly through the experiments which are shown later in the paper.

<ins>Understanding the Complexity of the problem</ins>

The difficulty can be seen through trying to solve this problem using the most naïve algorithm which the brute force method where every tour is considered.  The total possible tours can be understood by finding the number of unique permutations of ordering the nodes and then dividing by two. This is because a permutation represents the order in which the nodes can be visited and this number is divided by 2 since reversing the order will return a tour with the same cost. This is illustrated in the picture below.

*the distance is the same between arrows with the same number of marks*

Therefore, the number of possible tours can be given by the following equation, where $n$ is the number of nodes
$$
Total \,\,tours = \frac{(n-1)!}{2}
$$

If the $n$ factorial is not convincing enough, substituting numbers shows how much the complexity increases as the number of nodes increases.

| $n$ | $\frac{(n-1)!}{2}$ |
|---|---|
| 3 | 1 |
| 4 | 3 |
| 5 | 12 |
| 6 | 60 |
| 7 | 360 |
| 8 | 2520 |
| 9 | 20160 |
| 10 | 181440 |
| 11 | 1814400 |

## Pseudo-Code

Note in all these problems the start node is assumed to be the first node. In other implementations a start node can be specified. This may help decrease the search space depending on the algorithm but technically this should not matter since each tour must visit each node. 

#### <ins>Brute Force Algorithm</ins>
The brute force method was eluded to in the previous section. The pseudo code is as follows:
```text
Algorithm: Brute Force
Input: Fully Connected Graph
Output: Minimum Tour and cost of tour

F(graph)
	# from graph obtain nodes
	nodes = graph.nodes_list
	
	# define recursive function to find permutations of all nodes
	def recursive_function(current_tour, remaining_nodes, solution)
		# Base case
		if remaining == None:
			if reversed not in solution
				Add to solution
	
		# Recursive case
		for remaining nodes:
			recursive_function(current_tour + new_node, remaining node - new_node, solution)
		
	# Initialize loop
	recursive_function(nodes[0], nodes - nodes[0], solution_list)
	
	# Parse through solution list to find minimum weight (linear time)
	return min(solution)
```
To paraphrase the method behind this algorithm is to find every single permutation and then find the minimum cost out of all the found solutions. The time and space complexity of this algorithm are both $\mathcal{O}(n!)$. 

#### <ins>Dynamic Programming Algorithm</ins>

In this algorithm while creating the permutations the subgraph formed by the partial tour is saved such that they don't need to be traversed again. When exploring a new node two pieces of data is required to index the memoized data: the set of visited nodes and the index of the last visited node which represents the order. From this we can calculate the space complexity, which is $\mathcal{O}(n2^{n})$. This number represents the size of the matrix need to store the number of subproblems, which in this case is the weight of the subpaths and the final path. This number is derived from the fact that  each tour has to visit $n$ nodes and there are $2^{n}$ possible subsets. One way to understand the $2^{n}$ is by imagining representing the subset of selected nodes as a binary numeral, where 0 denotes the unvisited nodes and 1 denotes the visited ones. This concept is important to understand as it is the basis of the implementation. The time complexity is $\mathcal{O}(n^{2}2^{n})$ since each subproblem takes linear time to solve. It must be said that this implementation is heavily influenced by 

```text
Algorithm: Dynammic Programming
Input: Fully Connected Graph
Output: Minimum Tour and cost of tour
F(graph):
	# note start node is assumed to be the first node in list
	start = graph.nodes_list[0]
	
	# get graph lenght from adjacency matrix (or length of nodes list)
	length = size(graph.adjacency_matrix)

	# initialize 2D matrix for memoization
	store = matrix[length][2^length]

	# update store for first node
	update_store()
	
	# solve for the remaining graph using stored data
	solve()

	# solve for min cost using memoized data and find optimal tour
	cost = find_min_cost()
	tour = find_tour

	# store starting data in matrix
	def update_store(graph, store, length)
		for i in range(node):
			if node = start: 
				continue
			store[i][binary representation*] = graph.get_weights[start, node]
	
	def solve_subproblem(graph, store, length)
		# Note loop starts from 3 since first edge considers 2 nodes, so the
		  next iteration is the third node
		for r in range(3, N):
			# use another function to obtain every combination with one more
			  node than current state
			for subset in combinations(r, N):
				# skip if start is not in subset
				if start not in subset:
					continue
				# consider all nodes 	
				for next in all nodes:
					# skip of next is the start node or next is not in subset
					if next = start or next not in subset:
						continue
					# need to find best state using memoized data
					  so first take away next from subset using bit
					  manupulation
					state = subset - next 
				
	def find_min_cost(graph, store, length)
		
	def find_tour(graph, store, length)

	def combinations(r, N)
		return every combination state from r to N
	
```


### <ins>Christofides-Serdyukov Algorithm</ins>

```text
Algorithm: Christofides-Serdyukov
Input: Fully Connected Graph
Output: Minimum Tour and cost of tour
```

## Code Implementation

## Reflections

Sources
1 - Discrete math and its application 7th edition, rosen, pg 714
2 - https://www14.in.tum.de/personen/khan/Arindam%20Khan_files/2.%20metric%20TSP.pdf
3 - Nonlinear Optimization Bertsekas pg 637.
4 - Discrete math and its application 7th edition, rosen, pg 715
5 - https://www.math.uwaterloo.ca/tsp/history/index.html
6 - https://www.geeksforgeeks.org/proof-that-traveling-salesman-problem-is-np-hard/
7 - https://cdn.intechopen.com/pdfs/12736/intechtraveling_salesman_problem_an_overview_of_applications_formulations_and_solution_approaches.pdf

