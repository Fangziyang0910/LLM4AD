template_program = '''
import numpy as np


def aggregate_neighbors(node_features: np.ndarray, adj_matrix: np.ndarray, iteration: int) -> np.ndarray:
    """Aggregate features from neighboring nodes for one GNN layer.

    Args:
        node_features: Current node feature matrix with shape (n_nodes, n_features).
        adj_matrix: Binary symmetric adjacency matrix with shape (n_nodes, n_nodes).
        iteration: Current GNN layer index, starting at 0.

    Returns:
        Updated node feature matrix with shape (n_nodes, n_features).
    """
    degree = adj_matrix.sum(axis=1, keepdims=True)
    degree = np.where(degree == 0, 1.0, degree)
    return adj_matrix @ node_features / degree
'''

task_description = (
    "Design a neighborhood aggregation function for a Graph Neural Network applied "
    "to node classification on graphs with community structure. The evaluator applies "
    "the aggregation function for 3 GNN layers on fixed Stochastic Block Model graph "
    "instances. Initial node features contain a weak community signal buried in Gaussian "
    "noise. The final node representations are clustered with k-means, and performance "
    "is measured by the negative node classification error. Good aggregation should "
    "amplify within-community signal, suppress noise, and avoid excessive over-smoothing."
)
