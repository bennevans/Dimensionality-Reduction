

import numpy as np

def create_pairwise_distances(data, reduced_data):
    num_points = data.shape[0]
    pairwise_distances = np.zeros((num_points, num_points))
    pairwise_distances_reduced = np.zeros((num_points, num_points))
    for i, (datum_1, reduced_datum_1) in enumerate(zip(data, reduced_data)):
        for j, (datum_2, reduced_datum_2) in enumerate(zip(data, reduced_data)):
            dist = np.linalg.norm(datum_1 - datum_2)
            reduced_dist = np.linalg.norm(reduced_datum_1 - reduced_datum_2)
            pairwise_distances[i][j] = dist
            pairwise_distances_reduced[i][j] = reduced_dist

    return pairwise_distances, pairwise_distances_reduced    

def average_absolute_epsilon_from_pairwise_distances(pairwise_distances, pairwise_distances_reduced):
    epsilons = pairwise_distances_reduced / pairwise_distances - 1
    np.fill_diagonal(epsilons, 0)
    average_absolute_epsilon = np.average(np.abs(epsilons))
    return average_absolute_epsilon

def average_absolute_epsilon(data, reduced_data):
    pairwise_distances, pairwise_distances_reduced = create_pairwise_distances(data, reduced_data)
    return average_absolute_epsilon_from_pairwise_distances(pairwise_distances, pairwise_distances_reduced)
    
def get_epsilons(pairwise_distances, pairwise_distances_reduced):
    epsilons = pairwise_distances_reduced / pairwise_distances - 1
    np.fill_diagonal(epsilons, 0)
    indices = np.triu_indices(epsilons.shape[0])
    return epsilons[indices]

