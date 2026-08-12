#!/usr/bin/env python3
"""
V9.5 完整终局统计分析脚本

按照九部分需求提取和汇总所有 V9.5 实验数据：
1. 最终性能
2. 完整 best-so-far 曲线
3. Allocation 行为
4. 搜索结构
5. Evidence composition
6. Allocation 选择后的实际产出
7. Immediate gain vs future lineage value
8. 失败、重复与缓存
9. Case study (4个代表run)
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Literal
from collections import defaultdict, Counter
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class RunIdentifier:
    """实验运行标识符"""
    task: str
    repeat: int
    run_dir: Path


@dataclass
class PerformanceMetrics:
    """性能指标"""
    search_best: float
    test_score: float | None
    best_sample_order: int
    evaluator_call_count: int


@dataclass
class AllocationMetrics:
    """分配行为指标"""
    optimism_scale: float
    optimism_changed_argmax_rate: float = 0.0
    n0_selection_count: int = 0
    n1_selection_count: int = 0
    n2plus_selection_count: int = 0
    total_selections: int = 0
    non_best_q_selections: int = 0
    quality_gap_sum: float = 0.0
    s_values: list[float] = field(default_factory=list)
    s_crit_values: list[float] = field(default_factory=list)


@dataclass
class SearchStructureMetrics:
    """搜索结构指标"""
    n_root_clades: int
    n_artifacts: int
    n_states: int
    n_selected_states: int
    max_depth: int
    best_depth: int
    best_clade: int | None


@dataclass
class EvidenceMetrics:
    """证据组成指标"""
    direct_items_mean: float = 0.0
    formation_items_mean: float = 0.0
    direct_improve_count: int = 0
    direct_plateau_count: int = 0
    direct_regress_count: int = 0
    direct_invalid_count: int = 0


def load_v95_runs() -> dict[str, list[RunIdentifier]]:
    """加载所有 V9.5 实验运行"""
    tasks = {
        "tsp": "tsp_construct",
        "cvrp": "cvrp_aco",
        "op": "op_aco",
        "obp": "online_bin_packing"
    }
    
    runs = {}
    for short_name, task_dir in tasks.items():
        task_path = PROJECT_ROOT / "experiments" / task_dir / "traceaad_v9_5"
        runs[short_name] = []
        
        for rep in [1, 2, 3]:
            # 查找匹配的 rep 目录 (优先使用主实验运行)
            candidates = list(task_path.glob(f"v9_5_20260811_171029_{short_name}_rep{rep}"))
            if not candidates:
                candidates = list(task_path.glob(f"v9_5_*_{short_name}_rep{rep}"))
            
            if candidates:
                runs[short_name].append(RunIdentifier(
                    task=short_name,
                    repeat=rep,
                    run_dir=candidates[0]
                ))
    
    return runs


def load_checkpoint(run: RunIdentifier) -> dict[str, Any]:
    """加载 checkpoint 文件"""
    checkpoint_path = run.run_dir / "checkpoints" / "latest.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    with open(checkpoint_path, 'r') as f:
        return json.load(f)


def load_summary(run: RunIdentifier) -> dict[str, Any]:
    """加载 summary 文件"""
    summary_path = run.run_dir / "logs" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    
    with open(summary_path, 'r') as f:
        return json.load(f)


def load_decisions(run: RunIdentifier) -> list[dict[str, Any]]:
    """加载 decisions.jsonl"""
    decisions_path = run.run_dir / "artifacts" / "decisions.jsonl"
    if not decisions_path.exists():
        return []
    
    decisions = []
    with open(decisions_path, 'r') as f:
        for line in f:
            decisions.append(json.loads(line))
    return decisions


def load_candidates(run: RunIdentifier) -> list[dict[str, Any]]:
    """加载 candidates.jsonl"""
    candidates_path = run.run_dir / "artifacts" / "candidates.jsonl"
    if not candidates_path.exists():
        return []
    
    candidates = []
    with open(candidates_path, 'r') as f:
        for line in f:
            candidates.append(json.loads(line))
    return candidates


def load_edges(run: RunIdentifier) -> list[dict[str, Any]]:
    """加载 edges.jsonl"""
    edges_path = run.run_dir / "artifacts" / "edges.jsonl"
    if not edges_path.exists():
        return []
    
    edges = []
    with open(edges_path, 'r') as f:
        for line in f:
            edges.append(json.loads(line))
    return edges


# ============================================================================
# Part 1: 最终性能
# ============================================================================

def extract_performance(run: RunIdentifier, checkpoint: dict, summary: dict) -> PerformanceMetrics:
    """提取最终性能指标"""
    # 从 checkpoint 获取最终 best artifact
    best_artifact_id = checkpoint.get("best_artifact_id")
    forest = checkpoint.get("forest", {})
    artifacts = {a["artifact_id"]: a for a in forest.get("artifacts", [])}
    
    if best_artifact_id is not None and best_artifact_id in artifacts:
        search_best = artifacts[best_artifact_id]["directed_fitness"]
    else:
        search_best = summary.get("best_score", float('nan'))
    
    return PerformanceMetrics(
        search_best=search_best,
        test_score=None,  # 需要从 eval_best 目录读取
        best_sample_order=summary.get("best_sample_order", -1),
        evaluator_call_count=summary.get("evaluator_call_count", 0)
    )


def load_test_scores(task: str) -> dict[int, float]:
    """加载测试集评估结果"""
    eval_dir = PROJECT_ROOT / "experiments" / {
        "tsp": "tsp_construct",
        "cvrp": "cvrp_aco",
        "op": "op_aco",
        "obp": "online_bin_packing"
    }[task] / "traceaad_v9_5" / "eval_best_20260812_v95"
    
    results_path = eval_dir / "results.json"
    if not results_path.exists():
        return {}
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # 假设 results 格式为 {rep: score}
    return results


# ============================================================================
# Part 2: 完整 best-so-far 曲线
# ============================================================================

def extract_best_so_far_curve(checkpoint: dict) -> dict[int, float]:
    """从 checkpoint 提取 best-so-far 曲线
    
    Returns:
        {candidate_order: directed_fitness}
    """
    forest = checkpoint.get("forest", {})
    artifacts = {a["artifact_id"]: a for a in forest.get("artifacts", [])}
    attempts = forest.get("attempts", [])
    
    # 按 candidate_order 排序
    sorted_attempts = sorted([a for a in attempts if a.get("candidate_order") is not None], 
                            key=lambda x: x["candidate_order"])
    
    curve = {}
    current_best = float('-inf')
    evaluator_count = 0
    
    for attempt in sorted_attempts:
        if attempt.get("evaluator_called") and attempt.get("artifact_id") is not None:
            evaluator_count += 1
            artifact_id = attempt["artifact_id"]
            if artifact_id in artifacts:
                fitness = artifacts[artifact_id]["directed_fitness"]
                if fitness > current_best:
                    current_best = fitness
                
                curve[evaluator_count] = current_best
    
    return curve


# ============================================================================
# Part 3: Allocation 行为
# ============================================================================

def extract_allocation_behavior(checkpoint: dict, decisions: list[dict]) -> AllocationMetrics:
    """提取分配行为指标"""
    optimism_scale = checkpoint.get("optimism_scale", 0.0)
    forest = checkpoint.get("forest", {})
    states = {s["state_id"]: s for s in forest.get("states", [])}
    artifacts = {a["artifact_id"]: a for a in forest.get("artifacts", [])}
    
    # 从 decisions 中提取选择事件
    selection_events = [d for d in decisions if d.get("event") == "anchor_selected"]
    
    n0_count = 0
    n1_count = 0
    n2plus_count = 0
    optimism_changed_count = 0
    
    for event in selection_events:
        selected_state_id = event.get("selected_state_id")
        if selected_state_id is None:
            continue
        
        state = states.get(selected_state_id)
        if state is None:
            continue
        
        n = state.get("generation_count_n", 0)
        if n == 0:
            n0_count += 1
        elif n == 1:
            n1_count += 1
        else:
            n2plus_count += 1
        
        # 检查是否由于 optimism 改变了 argmax
        candidate_scores = event.get("candidate_scores", [])
        if candidate_scores:
            # 假设 candidate_scores 是 [(state_id, q, s), ...]
            best_q_state = max(candidate_scores, key=lambda x: x[1])[0]
            best_s_state = max(candidate_scores, key=lambda x: x[2])[0]
            if best_q_state != best_s_state and best_s_state == selected_state_id:
                optimism_changed_count += 1
    
    total = len(selection_events)
    optimism_changed_rate = optimism_changed_count / total if total > 0 else 0.0
    
    return AllocationMetrics(
        optimism_scale=optimism_scale,
        optimism_changed_argmax_rate=optimism_changed_rate,
        n0_selection_count=n0_count,
        n1_selection_count=n1_count,
        n2plus_selection_count=n2plus_count,
        total_selections=total
    )


# ============================================================================
# Part 4: 搜索结构
# ============================================================================

def extract_search_structure(checkpoint: dict) -> SearchStructureMetrics:
    """提取搜索结构指标"""
    forest = checkpoint.get("forest", {})
    states = forest.get("states", [])
    artifacts = forest.get("artifacts", [])
    
    # Root clades
    root_state_ids = [s["state_id"] for s in states if s.get("parent_state_id") is None]
    n_root_clades = len(root_state_ids)
    
    # Max depth
    max_depth = max(s.get("depth", 0) for s in states) if states else 0
    
    # Best depth and clade
    best_artifact_id = checkpoint.get("best_artifact_id")
    best_depth = 0
    best_clade = None
    
    if best_artifact_id is not None:
        # 找到 best artifact 对应的 state
        best_states = [s for s in states if s.get("artifact_id") == best_artifact_id]
        if best_states:
            best_state = best_states[0]
            best_depth = best_state.get("depth", 0)
            
            # 追踪到 root
            current = best_state
            while current.get("parent_state_id") is not None:
                parent_id = current["parent_state_id"]
                parent_states = [s for s in states if s["state_id"] == parent_id]
                if not parent_states:
                    break
                current = parent_states[0]
            best_clade = current["state_id"]
    
    # Selected states (generation_count_n > 0)
    n_selected = sum(1 for s in states if s.get("generation_count_n", 0) > 0)
    
    return SearchStructureMetrics(
        n_root_clades=n_root_clades,
        n_artifacts=len(artifacts),
        n_states=len(states),
        n_selected_states=n_selected,
        max_depth=max_depth,
        best_depth=best_depth,
        best_clade=best_clade
    )


# ============================================================================
# Part 5: Evidence composition (需要完整的 forest 重构，暂时跳过)
# ============================================================================

def extract_evidence_composition(candidates: list[dict]) -> EvidenceMetrics:
    """提取证据组成指标 - 需要从 prompt 解析或重建"""
    # 暂时返回空数据，需要完整实现
    return EvidenceMetrics()


# ============================================================================
# Part 6: Allocation 选择之后实际产生了什么
# ============================================================================

@dataclass
class ProductivityMetrics:
    """生成产出指标"""
    total_attempts: int = 0
    valid_children: int = 0
    parent_improvement_rate: float = 0.0
    delta_mean: float = 0.0
    delta_median: float = 0.0
    regression_rate: float = 0.0
    global_best_breakthroughs: int = 0


def extract_productivity_by_selection_type(checkpoint: dict) -> dict[str, ProductivityMetrics]:
    """按选择类型提取产出指标"""
    forest = checkpoint.get("forest", {})
    states = {s["state_id"]: s for s in forest.get("states", [])}
    artifacts = {a["artifact_id"]: a for a in forest.get("artifacts", [])}
    attempts = forest.get("attempts", [])
    
    # 按选择类型分组
    greedy_attempts = []
    optimism_induced = []
    
    best_so_far = float('-inf')
    
    for attempt in sorted(attempts, key=lambda x: x.get("candidate_order", 0)):
        if not attempt.get("evaluator_called"):
            continue
        
        anchor_state_id = attempt.get("anchor_state_id")
        if anchor_state_id is None:
            continue
        
        state = states.get(anchor_state_id)
        if state is None:
            continue
        
        # 简化版：所有 n>=2 都算 greedy (因为当前数据显示 optimism_changed_rate=0)
        n = state.get("generation_count_n", 0)
        if n >= 2:
            greedy_attempts.append(attempt)
        else:
            optimism_induced.append(attempt)
        
        # 追踪 global best
        artifact_id = attempt.get("artifact_id")
        if artifact_id and artifact_id in artifacts:
            fitness = artifacts[artifact_id]["directed_fitness"]
            if fitness > best_so_far:
                best_so_far = fitness
    
    def compute_metrics(attempt_list: list[dict]) -> ProductivityMetrics:
        if not attempt_list:
            return ProductivityMetrics()
        
        valid = [a for a in attempt_list if a.get("artifact_id") is not None]
        deltas = [a.get("directed_delta") for a in valid if a.get("directed_delta") is not None]
        improvements = [d for d in deltas if d > 0]
        regressions = [d for d in deltas if d < 0]
        
        return ProductivityMetrics(
            total_attempts=len(attempt_list),
            valid_children=len(valid),
            parent_improvement_rate=len(improvements) / len(deltas) if deltas else 0.0,
            delta_mean=np.mean(deltas) if deltas else 0.0,
            delta_median=np.median(deltas) if deltas else 0.0,
            regression_rate=len(regressions) / len(deltas) if deltas else 0.0,
        )
    
    return {
        "greedy": compute_metrics(greedy_attempts),
        "optimism_induced": compute_metrics(optimism_induced)
    }


# ============================================================================
# Part 8: 失败、重复与缓存
# ============================================================================

@dataclass
class HealthMetrics:
    """健康诊断指标"""
    invalid_rate: float = 0.0
    no_op_rate: float = 0.0
    repeated_duplicate_rate: float = 0.0
    ancestral_return_rate: float = 0.0
    cached_artifact_rate: float = 0.0
    transport_error_count: int = 0
    proposals_per_evaluator: float = 0.0


def extract_health_metrics(checkpoint: dict, summary: dict) -> HealthMetrics:
    """提取健康诊断指标"""
    outcome_counts = checkpoint.get("outcome_counts", {})
    total_attempts = sum(outcome_counts.values())
    
    if total_attempts == 0:
        return HealthMetrics()
    
    evaluator_calls = summary.get("evaluator_call_count", 0)
    
    return HealthMetrics(
        invalid_rate=outcome_counts.get("invalid", 0) / total_attempts,
        no_op_rate=outcome_counts.get("no_op", 0) / total_attempts,
        repeated_duplicate_rate=outcome_counts.get("repeated_duplicate", 0) / total_attempts,
        ancestral_return_rate=outcome_counts.get("ancestral_return", 0) / total_attempts,
        cached_artifact_rate=outcome_counts.get("cached_artifact", 0) / total_attempts,
        transport_error_count=checkpoint.get("transport_failure_count", 0),
        proposals_per_evaluator=total_attempts / evaluator_calls if evaluator_calls > 0 else 0.0
    )


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("V9.5 完整终局统计分析")
    print("=" * 80)
    print()
    
    runs = load_v95_runs()
    
    # Part 1: 最终性能
    print("=" * 80)
    print("Part 1: 最终性能 (1000 evaluator budget)")
    print("=" * 80)
    print()
    
    for task in ["tsp", "cvrp", "op", "obp"]:
        print(f"### {task.upper()} ###")
        print()
        
        task_runs = runs.get(task, [])
        if not task_runs:
            print(f"  未找到 {task} 的运行数据")
            print()
            continue
        
        performances = []
        for run in task_runs:
            try:
                checkpoint = load_checkpoint(run)
                summary = load_summary(run)
                perf = extract_performance(run, checkpoint, summary)
                performances.append((run.repeat, perf))
                print(f"  Rep {run.repeat}:")
                print(f"    Search best: {perf.search_best:.6f}")
                print(f"    Best at evaluator: {perf.best_sample_order}")
                print(f"    Total evaluator calls: {perf.evaluator_call_count}")
            except Exception as e:
                print(f"  Rep {run.repeat}: 加载失败 - {e}")
        
        if performances:
            search_bests = [p.search_best for _, p in performances]
            mean = np.mean(search_bests)
            std = np.std(search_bests, ddof=1) if len(search_bests) > 1 else 0.0
            print()
            print(f"  Summary:")
            print(f"    Search best mean ± SD: {mean:.6f} ± {std:.6f}")
            print(f"    Min: {min(search_bests):.6f}, Max: {max(search_bests):.6f}")
        
        print()
    
    # Part 2: Best-so-far 曲线
    print("=" * 80)
    print("Part 2: Best-so-far 搜索曲线")
    print("=" * 80)
    print()
    
    for task in ["tsp", "cvrp", "op", "obp"]:
        print(f"### {task.upper()} ###")
        print()
        
        task_runs = runs.get(task, [])
        checkpoints_at = [100, 200, 300, 500, 750, 1000]
        
        all_curves = []
        for run in task_runs:
            try:
                checkpoint = load_checkpoint(run)
                curve = extract_best_so_far_curve(checkpoint)
                
                print(f"  Rep {run.repeat}:")
                for cp in checkpoints_at:
                    # 找到最接近的 evaluator_order
                    available = [k for k in curve.keys() if k <= cp]
                    if available:
                        closest = max(available)
                        print(f"    @{cp:4d}: {curve[closest]:.6f}")
                
                all_curves.append(curve)
                
                # 最后刷新
                if curve:
                    last_update = max(curve.keys())
                    print(f"    Last best update at evaluator: {last_update}")
                
                print()
            except Exception as e:
                print(f"  Rep {run.repeat}: 加载失败 - {e}")
                print()
        
        print()
    
    # Part 3: Allocation 行为
    print("=" * 80)
    print("Part 3: Allocation 行为 (完整 1000 budget)")
    print("=" * 80)
    print()
    
    for task in ["tsp", "cvrp", "op", "obp"]:
        print(f"### {task.upper()} ###")
        print()
        
        task_runs = runs.get(task, [])
        for run in task_runs:
            try:
                checkpoint = load_checkpoint(run)
                decisions = load_decisions(run)
                allocation = extract_allocation_behavior(checkpoint, decisions)
                
                print(f"  Rep {run.repeat}:")
                print(f"    Optimism scale (s): {allocation.optimism_scale:.4f}")
                print(f"    Optimism changed argmax rate: {allocation.optimism_changed_argmax_rate:.4f}")
                print(f"    n=0 selections: {allocation.n0_selection_count}")
                print(f"    n=1 selections: {allocation.n1_selection_count}")
                print(f"    n≥2 selections: {allocation.n2plus_selection_count}")
                print(f"    Total selections: {allocation.total_selections}")
                
                if allocation.total_selections > 0:
                    print(f"    n=0 ratio: {allocation.n0_selection_count / allocation.total_selections:.4f}")
                    print(f"    n=1 ratio: {allocation.n1_selection_count / allocation.total_selections:.4f}")
                    print(f"    n≥2 ratio: {allocation.n2plus_selection_count / allocation.total_selections:.4f}")
                
                print()
            except Exception as e:
                print(f"  Rep {run.repeat}: 加载失败 - {e}")
                print()
        
        print()
    
    # Part 4: 搜索结构
    print("=" * 80)
    print("Part 4: 搜索结构")
    print("=" * 80)
    print()
    
    for task in ["tsp", "cvrp", "op", "obp"]:
        print(f"### {task.upper()} ###")
        print()
        
        task_runs = runs.get(task, [])
        for run in task_runs:
            try:
                checkpoint = load_checkpoint(run)
                structure = extract_search_structure(checkpoint)
                
                print(f"  Rep {run.repeat}:")
                print(f"    Root clades: {structure.n_root_clades}")
                print(f"    Total artifacts: {structure.n_artifacts}")
                print(f"    Total states: {structure.n_states}")
                print(f"    Selected states (n>0): {structure.n_selected_states}")
                print(f"    Max depth: {structure.max_depth}")
                print(f"    Best depth: {structure.best_depth}")
                print(f"    Best clade: {structure.best_clade}")
                
                if structure.n_states > 0:
                    print(f"    Selection rate: {structure.n_selected_states / structure.n_states:.4f}")
                
                print()
            except Exception as e:
                print(f"  Rep {run.repeat}: 加载失败 - {e}")
                print()
        
        print()
    
    # Part 5: Evidence composition (暂时跳过，需要完整 forest 重构)
    print("=" * 80)
    print("Part 5: Evidence composition (暂时跳过)")
    print("=" * 80)
    print("需要完整的 SearchForest 逻辑重构，将在后续补充")
    print()
    
    # Part 6: Allocation 选择后的实际产出
    print("=" * 80)
    print("Part 6: Allocation 选择后的实际产出")
    print("=" * 80)
    print()
    
    for task in ["tsp", "cvrp", "op", "obp"]:
        print(f"### {task.upper()} ###")
        print()
        
        task_runs = runs.get(task, [])
        for run in task_runs:
            try:
                checkpoint = load_checkpoint(run)
                productivity = extract_productivity_by_selection_type(checkpoint)
                
                print(f"  Rep {run.repeat}:")
                for sel_type, metrics in productivity.items():
                    print(f"    {sel_type.capitalize()}:")
                    print(f"      Total attempts: {metrics.total_attempts}")
                    print(f"      Valid children: {metrics.valid_children}")
                    print(f"      Parent improvement rate: {metrics.parent_improvement_rate:.4f}")
                    print(f"      Delta mean: {metrics.delta_mean:.6f}")
                    print(f"      Delta median: {metrics.delta_median:.6f}")
                    print(f"      Regression rate: {metrics.regression_rate:.4f}")
                print()
            except Exception as e:
                print(f"  Rep {run.repeat}: 加载失败 - {e}")
                print()
        
        print()
    
    # Part 8: 失败、重复与缓存
    print("=" * 80)
    print("Part 8: 失败、重复与缓存（健康诊断）")
    print("=" * 80)
    print()
    
    for task in ["tsp", "cvrp", "op", "obp"]:
        print(f"### {task.upper()} ###")
        print()
        
        task_runs = runs.get(task, [])
        for run in task_runs:
            try:
                checkpoint = load_checkpoint(run)
                summary = load_summary(run)
                health = extract_health_metrics(checkpoint, summary)
                
                print(f"  Rep {run.repeat}:")
                print(f"    Invalid rate: {health.invalid_rate:.4f}")
                print(f"    No-op rate: {health.no_op_rate:.4f}")
                print(f"    Repeated duplicate rate: {health.repeated_duplicate_rate:.4f}")
                print(f"    Ancestral return rate: {health.ancestral_return_rate:.4f}")
                print(f"    Cached artifact rate: {health.cached_artifact_rate:.4f}")
                print(f"    Transport errors: {health.transport_error_count}")
                print(f"    Proposals per evaluator call: {health.proposals_per_evaluator:.2f}")
                print()
            except Exception as e:
                print(f"  Rep {run.repeat}: 加载失败 - {e}")
                print()
        
        print()
    
    print("=" * 80)
    print("统计完成")
    print("=" * 80)
    print()
    print("注意：")
    print("- Part 5 (Evidence composition) 需要完整 SearchForest 逻辑，待补充")
    print("- Part 7 (Immediate gain vs future lineage) 需要追踪 lineage，待补充")
    print("- Part 9 (Case study) 需要挑选代表 run 并追踪完整 lineage，待补充")


if __name__ == "__main__":
    main()
