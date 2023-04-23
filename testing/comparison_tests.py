"""
comparison_test.py

The following script contains the functions for testing the 3 algorithms
for solving the travelling salesman problem: Brute Force, Dynamic
Programming and Christofides-Serdyukov Algorithm

Below is a list of functions in this file and a brief description of their
usage:

Generate graph - Function to create fully connected graph with random
 weights given number of nodes.

run_all - Function to run all three algorithms and collect their individual
run times per size of graph. The solutions are saved to a csv with filename
specified by user. Note the run times are calculated for graphs with 3 to
n nodes.

run_some - Function to run all three algorithms and collect their
individual run times per size of graph. The solutions are saved to a csv
with filename specified by user. Note the run times are calculated for
graphs with 3 to n nodes

compare_christofides_serdyukov_accuracy - Function to compare christofides-
serdyukov algorithm with optimal solution found by dynamic programming

"""

import time
import csv
import random
from utilities import graph_class
from brute_force import brute_force
from dynamic_programming import dynamic_programming
from christofides_serdyukov import christofides_serdyukov_algorithm


def generate_graph(n):
    """
    Function to create fully connected graph object
    :param n: number of nodes in graph
    :return: graph object
    """
    # Generate nodes
    nodes = [i for i in range(n)]

    # create edges and assign random weights
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j, random.randint(1, 2 * n)))

    # return graph
    return graph_class(nodes, edges)


def run_all(n, filename):
    """
    Function to run all algorithms and create csv file with run times of
    optimal tour of graphs of size 3 - n.
    :param n: maximum number of nodes
    :param filename: Output filename NOTE filename must have ".csv" at
                     the end
    :return: array of run times
    """

    # initialize array to store solutions
    times = []

    # loop through 3 to n
    for i in range(3, n):
        # create random fully connected graph
        test_graph = generate_graph(i)

        # calculate time for brute force algorithm
        start_brute_force = time.time()
        min_tour_edges, min_weight = brute_force(test_graph)
        end_brute_force = time.time()
        brute_force_time = end_brute_force - start_brute_force

        # calculate time for dynamic programming algorithm
        start_dynamic_programming = time.time()
        dp_obj = dynamic_programming(test_graph)
        end_dynamic_programming = time.time()
        dynamic_programming_time = end_dynamic_programming \
                                   - start_dynamic_programming

        # brute force and dynamic programming should have the same
        # solution
        if min_weight != dp_obj.min_tour_cost:
            print("Brute force and Dynamic programming does not return"
                  " the same solution!")

        # calculate time for christofides-serdyukov algorithm
        start_christofides_serdyukov = time.time()
        min_tour_edges, min_weight = christofides_serdyukov_algorithm(
            test_graph)
        end_christofides_serdyukov = time.time()
        christofides_serdyukov_time = end_christofides_serdyukov \
                                      - start_christofides_serdyukov

        # append times to array
        times.append([str(i), str(brute_force_time),
                      str(dynamic_programming_time),
                      str(christofides_serdyukov_time)])

    # create header
    table_header = ["n", "Brute Force", "Dynammic Programming",
                    "Christofides-Serdyukov"]

    # put into file
    with open(filename, 'w') as solution_file:
        write_to_csv = csv.writer(solution_file)
        write_to_csv.writerow(table_header)
        for individual_time in times:
            write_to_csv.writerow(individual_time)
    solution_file.close()
    return times


def run_some(n, functions_to_run, filename):
    """
    Function to run all algorithms and create csv file with run times of
    optimal tour of graphs of size 3 - n.
    :param n: maximum number of nodes
    :param functions_to_run: Binary array representing which algorithms to
                            run e.g. if only want to run brute force the
                            input would be [1, 0, 0]
    :param filename: Output filename NOTE filename must have ".csv" at
                     the end
    :return: array of run times
    """

    functions = [brute_force, dynamic_programming,
                 christofides_serdyukov_algorithm]

    # initialize array to store solutions
    times = []

    # loop through 3 to n
    for i in range(3, n):
        test_graph = generate_graph(i)
        iteration_time = []
        # loop through specified functions to run and store in solution
        # array
        for index, binary in enumerate(functions_to_run):
            if binary == 1:
                start = time.time()
                # run function
                _ = functions[index](test_graph)
                end = time.time()
                solution_time = end - start
                iteration_time.append(str(solution_time))

        times.append([str(i)] + iteration_time)
    # create table header
    table_header = ["n"]

    if functions_to_run[0] == 1:
        table_header += ["Brute Force"]

    if functions_to_run[1] == 1:
        table_header += ["Dynamic Programming"]

    if functions_to_run[2] == 1:
        table_header += ["Christofides-Serdyukov"]

    # store in output file
    with open(filename, 'w') as solution_file:
        write_to_csv = csv.writer(solution_file)
        write_to_csv.writerow(table_header)
        for individual_time in times:
            write_to_csv.writerow(individual_time)
    solution_file.close()
    return times


def compare_christofides_serdyukov_accuracy(n, filename):
    """
    Function to compare tour found using christofides-serdyukov algorithm

    :param n: maximum number of nodes
    :param filename: Output filename NOTE filename must have ".csv" at
                     the end
    :return: array of error_ratios
    """
    # initialize array to store solutions
    error_ratios = []

    # loop through 3 to n
    for i in range(3, n):
        test_graph = generate_graph(i)
        # calculate dynamic progam solution
        dp_obj = dynamic_programming(test_graph)
        # calculate christofides-serdyukov solution
        min_tour_edges, min_weight = christofides_serdyukov_algorithm(
            test_graph)

        # calculate error ratio and append to solution list
        error_ratio = min_weight / dp_obj.min_tour_cost
        error_ratios.append([str(i), error_ratio])

    # save to file
    table_header = ["n", "Percentage Difference"]
    with open(filename, 'w') as solution_file:
        write_to_csv = csv.writer(solution_file)
        write_to_csv.writerow(table_header)
        for individual_percentage in error_ratios:
            write_to_csv.writerow(individual_percentage)
    solution_file.close()

    return error_ratios
