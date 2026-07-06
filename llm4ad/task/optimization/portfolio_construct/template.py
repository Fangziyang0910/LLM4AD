template_program = '''
import numpy as np


def score_assets(asset_returns: np.ndarray,
                 selected_indices: np.ndarray,
                 candidate_indices: np.ndarray) -> np.ndarray:
    """Score candidate assets for greedy portfolio construction.

    Args:
        asset_returns: historical daily returns with shape (n_assets, n_periods).
        selected_indices: already selected asset indices.
        candidate_indices: candidate asset indices to score.

    Returns:
        Score array with length len(candidate_indices). Higher is better.
    """
    scores = np.array([
        asset_returns[i].mean() / (asset_returns[i].std() + 1e-8)
        for i in candidate_indices
    ])
    return scores
'''

task_description = (
    "Design an asset scoring function for greedy portfolio construction. At each "
    "selection step the evaluator calls the function with all historical asset returns, "
    "the already selected assets, and remaining candidates, then adds the highest-scoring "
    "candidate to an equal-weighted portfolio. The score should balance expected return, "
    "volatility, and diversification relative to selected assets. Performance is measured "
    "by annualized Sharpe ratio on fixed synthetic one-factor return instances."
)
