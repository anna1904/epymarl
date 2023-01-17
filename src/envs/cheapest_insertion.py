import numpy as np

def cheapest_insertion(tour, node):
    min_cost = 1000
    min_index = 0
    for i in range(len(tour)-1):
        dist_prev = np.linalg.norm(tour[i] - tour[i+1])
        dist_future = np.linalg.norm(tour[i] - node) + np.linalg.norm(tour[i+1] - node)
        if (dist_future - dist_prev) < min_cost:
            min_cost = dist_future - dist_prev
            min_index = i
    updated_route = np.insert(tour, min_index+1, node, axis=0)

    return updated_route, min_cost, min_index