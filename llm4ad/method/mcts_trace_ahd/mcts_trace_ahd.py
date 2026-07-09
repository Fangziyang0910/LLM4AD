from __future__ import annotations

from llm4ad.method.mcts_ahd.mcts import MCTS, MCTSNode
from llm4ad.method.mcts_ahd.mcts_ahd import MCTS_AHD

from .prompt import MCTSTracePrompt


class MCTS_Trace_AHD(MCTS_AHD):
    """MCTS-AHD variant that injects trace guidance into all expansion operators."""

    def _ordered_trace_from_node(self, cur_node: MCTSNode):
        trace = []
        now = cur_node
        while now is not None and now.algorithm != "Root":
            if getattr(now, "individual", None) is not None:
                trace.append(now.individual)
            now = now.parent
        return tuple(reversed(trace))

    def expand(self, mcts: MCTS, node_set, cur_node: MCTSNode, option: str):
        if getattr(self, '_search_aborted', False):
            return node_set
        is_valid_func = True
        trace_indivs = self._ordered_trace_from_node(cur_node)

        if option == 's1':
            if len(trace_indivs) == 1:
                return node_set

            i = 0
            while i < 3:
                prompt = MCTSTracePrompt.get_prompt_s1_trace(
                    self._task_description_str,
                    trace_indivs,
                    cur_node.individual,
                    self._function_to_evolve,
                )
                func = self._sample_evaluate_register(prompt, func_only=True, operator=option)
                if func is False:
                    is_valid_func = False
                    i += 1
                    continue
                is_valid_func = (func.score is not None) and not self.check_duplicate(node_set, str(func))
                if is_valid_func is False:
                    i += 1
                    continue
                else:
                    break

        elif option == 'e1':
            indivs = self._sample_e1_references_from_root(mcts)
            if len(indivs) == 0:
                return node_set
            prompt = MCTSTracePrompt.get_prompt_e1_trace(self._task_description_str, indivs, self._function_to_evolve)
            func = self._sample_evaluate_register(prompt, func_only=True, operator=option)
            if func is False:
                is_valid_func = False
            else:
                is_valid_func = (func.score is not None)

        elif option == 'e2':
            i = 0
            while i < 3:
                elite_set = [
                    individual for individual in self._current_elite_set()
                    if individual != cur_node.individual
                ]
                if len(elite_set) == 0:
                    return node_set
                now_indiv = self._population.selection(elite_set)
                prompt = MCTSTracePrompt.get_prompt_e2_trace(
                    self._task_description_str,
                    [now_indiv, cur_node.individual],
                    self._function_to_evolve,
                    trace_indivs,
                )
                func = self._sample_evaluate_register(prompt, func_only=True, operator=option)
                if func is False:
                    is_valid_func = False
                    i += 1
                    continue
                is_valid_func = (func.score is not None) and not self.check_duplicate(node_set, str(func))
                if is_valid_func is False:
                    i += 1
                    continue
                else:
                    break

        elif option == 'm1':
            i = 0
            while i < 3:
                prompt = MCTSTracePrompt.get_prompt_m1_trace(
                    self._task_description_str,
                    cur_node.individual,
                    self._function_to_evolve,
                    trace_indivs,
                )
                func = self._sample_evaluate_register(prompt, func_only=True, operator=option)
                if func is False:
                    is_valid_func = False
                    i += 1
                    continue
                is_valid_func = (func.score is not None) and not self.check_duplicate(node_set, str(func))
                if is_valid_func is False:
                    i += 1
                    continue
                else:
                    break

        elif option == 'm2':
            i = 0
            while i < 3:
                prompt = MCTSTracePrompt.get_prompt_m2_trace(
                    self._task_description_str,
                    cur_node.individual,
                    self._function_to_evolve,
                    trace_indivs,
                )
                func = self._sample_evaluate_register(prompt, func_only=True, operator=option)
                if func is False:
                    is_valid_func = False
                    i += 1
                    continue
                is_valid_func = (func.score is not None) and not self.check_duplicate(node_set, str(func))
                if is_valid_func is False:
                    i += 1
                    continue
                else:
                    break

        else:
            assert False, 'Invalid option!'

        if not is_valid_func:
            self._log_mcts_event(
                event='expand',
                status='invalid',
                reason='timeout_or_invalid_function',
                operator=option,
                sample_order=self._tot_sample_nums,
                parent_score=self._node_score(cur_node),
                parent_depth=cur_node.depth,
                parent_visits=cur_node.visits,
            )
            return node_set

        if option != 'e1':
            parent_score = self._node_score(cur_node)
        else:
            if self.check_duplicate_obj(node_set, func.score):
                self._log_mcts_event(
                    event='expand',
                    status='duplicate',
                    reason='duplicate_e1_objective',
                    operator=option,
                    sample_order=self._tot_sample_nums,
                    parent_score=None,
                    child_score=func.score,
                    parent_depth=cur_node.depth,
                    parent_visits=cur_node.visits,
                )
                return node_set
            parent_score = None

        if is_valid_func and func.score != float('-inf'):
            self._population.register_function(func)
            now_node = MCTSNode(func.algorithm, str(func), -1 * func.score, individual=func,
                                parent=cur_node, depth=cur_node.depth + 1, visit=1, Q=func.score, raw_info=func)
            if option == 'e1':
                now_node.subtree.append(now_node)
            cur_node.add_child(now_node)
            mcts.backpropagate(now_node)
            if node_set is not cur_node.children:
                node_set.append(now_node)
            self._log_mcts_event(
                event='expand',
                status='expanded',
                operator=option,
                sample_order=self._tot_sample_nums,
                parent_score=parent_score,
                child_score=func.score,
                parent_depth=cur_node.depth,
                child_depth=now_node.depth,
                parent_visits=cur_node.visits,
                child_visits=now_node.visits,
                parent_children_count=len(cur_node.children),
                root_parent=cur_node.algorithm == 'Root',
            )
        return node_set
