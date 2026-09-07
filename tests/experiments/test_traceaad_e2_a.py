from collections import defaultdict

import numpy as np

from experiments.traceaad_e2_a.analyze import hajek, interaction, quality_kernel, trajectory_state


def test_trajectory_state_excludes_current_action_and_direct_parent_from_revisit():
    nodes = {
        0: {'parent_id': None, 'fitness': 1.0, 'operator': 'Init'},
        1: {'parent_id': 0, 'fitness': 1.1, 'operator': 'Refine'},
        2: {'parent_id': None, 'fitness': 0.9, 'operator': 'Init'},
    }
    born = {0: 1, 1: 2, 2: 3}
    behavior = np.array([[0.0, 0.1, 0.8], [0.1, 0.0, 0.6], [0.8, 0.6, 0.0]])
    state = trajectory_state({'parent_id': 1, 'candidate_id': 4}, nodes, born, [0, 1, 2], behavior)
    assert state['move'] == 0.1
    assert state['revisit'] == 0.6
    assert np.isclose(state['recent_gain'], 0.1)


def test_hajek_interaction_uses_logged_binary_propensity():
    rows = []
    for stagnant, refine, pivot in [(False, 0.1, 0.2), (True, 0.1, 0.5)]:
        for action, value in [('Refine', refine), ('Pivot', pivot)]:
            rows += [{'stagnant': stagnant, 'action': action, 'pivot_propensity': 0.25,
                      'frontier_gain': value} for _ in range(3)]
    result = interaction(rows, 'frontier_gain')
    assert np.isclose(result['nonstagnant']['tau'], 0.1)
    assert np.isclose(result['stagnant']['tau'], 0.4)
    assert np.isclose(result['interaction'], 0.3)
    assert hajek(rows, 'frontier_gain', True)['Pivot']['ess'] == 3


def test_quality_kernel_excludes_current_parent_history():
    nodes = {0: {'fitness': 1.0}, 1: {'fitness': 1.1}, 2: {'fitness': 1.2}}
    history = defaultdict(lambda: [0, 0], {0: [100, 100], 1: [2, 0], 2: [2, 2]})
    prediction = quality_kernel(0, {0, 1, 2}, history, nodes, 0.1)
    history[0] = [100, 0]
    assert quality_kernel(0, {0, 1, 2}, history, nodes, 0.1) == prediction
