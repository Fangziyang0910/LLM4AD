"""计算各任务测试实例的参考最优值（精确或认证近优）。

- tsp_construct：CP-SAT AddCircuit 精确求解（距离 ×1e6 整数化）。
- op_aco：CP-SAT AddCircuit + 预算约束 + 奖品最大化（未访问节点用自环字面量）。
- cvrp_aco：CP-SAT 单一大回路 + depot 副本 + 容量链（精确尝试）；以及
  PyVRP(HGS) 近优求解（--solver pyvrp）。
- vrptw_construct：CP-SAT depot 副本 + 容量 + 时间窗传播；PyVRP 同上。
- online_bin_packing：在线问题无单一最优，输出 Martello-Toth L1/L2 下界、
  FFD 离线近优、First-Fit/Best-Fit 经典在线参考（与被评方法同序列）。

结果按实例写入 JSON、断点续跑（已存在的实例跳过）。

    uv run python experiments/analysis/compute_reference_optima.py --task tsp --sizes 50 --seed 2025
    uv run python experiments/analysis/compute_reference_optima.py --task cvrp --split test_50 --solver cpsat
    uv run python experiments/analysis/compute_reference_optima.py --task op --split test_50
    uv run python experiments/analysis/compute_reference_optima.py --task vrptw --seed 2025
    uv run python experiments/analysis/compute_reference_optima.py --task obp
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from bisect import bisect_left, insort
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from ortools.sat.python import cp_model

SCALE = 10**6


# ---------------------------------------------------------------------------
# CP-SAT 基础
# ---------------------------------------------------------------------------

def _solve(model: cp_model.CpModel, limit: float, workers: int) -> tuple[str, float | None, float | None, float]:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = limit
    solver.parameters.num_workers = workers
    started = time.time()
    status = solver.Solve(model)
    name = solver.StatusName(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return name, None, None, time.time() - started
    return name, solver.ObjectiveValue() / SCALE, solver.BestObjectiveBound() / SCALE, time.time() - started


def _int_dist(coordinates: np.ndarray) -> np.ndarray:
    diff = coordinates[:, None, :] - coordinates[None, :, :]
    return np.round(np.linalg.norm(diff, axis=2) * SCALE).astype(np.int64)


# ---------------------------------------------------------------------------
# TSP
# ---------------------------------------------------------------------------

def _tsp_row(args: tuple[np.ndarray, float, int]) -> dict:
    dist, limit, workers = args
    n = len(dist)
    di = np.round(dist * SCALE).astype(np.int64)
    model = cp_model.CpModel()
    arcs = {
        (i, j): model.NewBoolVar(f"x{i}_{j}")
        for i in range(n)
        for j in range(n)
        if i != j
    }
    model.AddCircuit([(i, j, var) for (i, j), var in arcs.items()])
    model.Minimize(sum(int(di[i][j]) * var for (i, j), var in arcs.items()))
    status, obj, bound, secs = _solve(model, limit, workers)
    return {"status": status, "objective": obj, "bound": bound, "seconds": secs}


def run_tsp(sizes: list[int], seed: int, limit: float, workers: int, jobs: int, out_dir: Path) -> None:
    from llm4ad.task.optimization.tsp_construct.get_instance import GetData

    for size in sizes:
        instances = GetData(n_instance=16, n_cities=size, seed=seed).generate_instances()
        out_path = out_dir / f"reference_optima_tsp{size}_seed{seed}.json"
        rows = _load_existing(out_path)
        payload = [
            (k, (dist, limit, workers))
            for k, (_coords, dist) in enumerate(instances)
            if str(k) not in rows or not _cpsat_resumable(rows[str(k)])
        ]
        _run_pool(payload, _tsp_row, rows, out_path, lambda row: row["status"] == "OPTIMAL", label=f"tsp{size}_seed{seed}", jobs=jobs)


# ---------------------------------------------------------------------------
# OP（奖品最大化，预算约束）
# ---------------------------------------------------------------------------

def _op_row(args: tuple[np.ndarray, float, np.ndarray, float, int]) -> dict:
    coordinates, prizes, max_len, limit, workers = args
    n = len(coordinates)
    di = _int_dist(coordinates)
    pi = np.round(np.asarray(prizes) * SCALE).astype(np.int64)
    model = cp_model.CpModel()
    visit = [model.NewBoolVar(f"v{i}") for i in range(n)]
    real_arcs: list[tuple[int, int, object]] = []
    circuit: list[tuple[int, int, object]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lit = model.NewBoolVar(f"x{i}_{j}")
            real_arcs.append((i, j, lit))
            circuit.append((i, j, lit))
    for i in range(1, n):
        circuit.append((i, i, visit[i].Not()))  # 自环 = 节点未被访问
    model.AddCircuit(circuit)
    budget = round(float(max_len) * SCALE)
    model.Add(sum(int(di[i][j]) * lit for i, j, lit in real_arcs) <= budget)
    model.Maximize(sum(int(pi[i]) * visit[i] for i in range(1, n)))
    status, obj, bound, secs = _solve(model, limit, workers)
    return {"status": status, "objective": obj, "bound": bound, "seconds": secs}


def run_op(split: str, limit: float, workers: int, jobs: int, out_dir: Path) -> None:
    from llm4ad.task.optimization.op_aco.dataset import gen_distance_matrix, gen_prizes, load_split_instances

    coordinates, metadata = load_split_instances(split)
    max_len = float(metadata["max_len"])
    out_path = out_dir / f"reference_optima_op_{split}.json"
    rows = _load_existing(out_path)
    payload = [
        (k, (coords, gen_prizes(coords), max_len, limit, workers))
        for k, coords in enumerate(coordinates)
        if str(k) not in rows or not _cpsat_resumable(rows[str(k)])
    ]
    _run_pool(payload, _op_row, rows, out_path, lambda row: row["status"] == "OPTIMAL", label=f"op_{split}", jobs=jobs)


# ---------------------------------------------------------------------------
# CVRP / VRPTW（depot 副本 + 单一大回路 + 容量/时间传播）
# ---------------------------------------------------------------------------

def _vehicle_model(
    distances: np.ndarray,
    demands: np.ndarray,
    capacity: int,
    k_copies: int,
    *,
    time_windows: np.ndarray | None = None,
    service: np.ndarray | None = None,
    horizon: float | None = None,
) -> cp_model.CpModel:
    """clients 1..n；depot 复制为节点 0..k-1。副本间弧代价 0。"""
    n_clients = len(demands)
    total = k_copies + n_clients
    di = np.round(np.asarray(distances, dtype=np.float64) * SCALE).astype(np.int64)
    expanded = np.zeros((total, total), dtype=np.int64)
    expanded[k_copies:, k_copies:] = di[1:, 1:]
    expanded[:k_copies, k_copies:] = di[0, 1:][None, :]
    expanded[k_copies:, :k_copies] = di[1:, 0][:, None]

    model = cp_model.CpModel()
    arcs = {}
    for a in range(total):
        for b in range(total):
            if a != b:
                arcs[a, b] = model.NewBoolVar(f"x{a}_{b}")
    model.AddCircuit([(a, b, var) for (a, b), var in arcs.items()])
    model.Minimize(sum(int(expanded[a][b]) * var for (a, b), var in arcs.items()))

    def client(node: int) -> int:
        return node - k_copies + 1  # 原始实例行号

    load = {}
    time = {}
    for node in range(k_copies, total):
        c = client(node)
        load[node] = model.NewIntVar(int(demands[c - 1]), int(capacity), f"load{node}")
        if time_windows is not None:
            lo = int(round(float(time_windows[c][0]) * SCALE))
            hi = int(round(float(time_windows[c][1]) * SCALE))
            time[node] = model.NewIntVar(lo, hi, f"t{node}")

    big_m = int(round(((horizon or 0.0) + math.sqrt(2.0) + 1.0) * SCALE)) if time_windows is not None else 0
    for (a, b), lit in arcs.items():
        if a >= k_copies:  # client -> *
            ca = client(a)
            if b >= k_copies:
                cb = client(b)
                model.Add(
                    load[b] >= load[a] + int(demands[cb - 1]) - int(capacity) * (1 - lit)
                )
                if time_windows is not None:
                    step = int(round((float(service[ca]) + float(distances[ca][cb])) * SCALE))
                    model.Add(time[b] >= time[a] + step - big_m * (1 - lit))
            elif time_windows is not None:
                pass  # 回到副本：时间不受限
        else:  # copy -> client
            if b >= k_copies and time_windows is not None:
                cb = client(b)
                step = int(round(float(distances[0][cb]) * SCALE))
                model.Add(time[b] >= step - big_m * (1 - lit))
    return model


def _cvrp_row(args: tuple[np.ndarray, int, float, int]) -> dict:
    instance, capacity, limit, workers = args
    coordinates = instance[:, 1:]
    demands = instance[1:, 0]
    distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
    n = len(demands)
    k = math.ceil(float(demands.sum()) / capacity) + 3
    attempts = []
    for _ in range(3):
        model = _vehicle_model(distances, demands, capacity, min(k, n + 2))
        status, obj, bound, secs = _solve(model, limit, workers)
        attempts.append({"k_copies": min(k, n + 2), "status": status, "objective": obj, "bound": bound, "seconds": secs})
        if status in ("OPTIMAL", "FEASIBLE"):
            break
        if status == "INFEASIBLE":
            k = min(k * 2, n + 2)
            continue
        break
    best = next((a for a in attempts if a["objective"] is not None), attempts[-1])
    return {**best, "attempts": attempts}


def run_cvrp(split: str, limit: float, workers: int, jobs: int, out_dir: Path) -> None:
    from llm4ad.task.optimization.cvrp_aco.dataset import load_split_instances

    instances, metadata = load_split_instances(split)
    capacity = int(metadata["capacity"])
    out_path = out_dir / f"reference_optima_cvrp_{split}.json"
    rows = _load_existing(out_path)
    payload = [
        (k, (instance, capacity, limit, workers))
        for k, instance in enumerate(instances)
        if str(k) not in rows or not _cpsat_resumable(rows[str(k)])
    ]
    _run_pool(payload, _cvrp_row, rows, out_path, lambda row: row["status"] == "OPTIMAL", label=f"cvrp_{split}", jobs=jobs)


def _vrptw_row(args: tuple[tuple, float, int]) -> dict:
    (coordinates, distances, demands, capacity, service, windows), limit, workers = args
    dem = np.asarray(demands[1:], dtype=np.float64)
    n = len(dem)
    # 车辆数上限必须宽松：时间窗可能迫使远超容量下界的路线数，K 过小会把
    # 模型限制在"至多 K 条路线"的子问题上，给出偏高的假最优（HGS 交叉验证
    # 曾在 8/32 个实例上优于受限最优，最大差 1.7%）。
    k = min(30, n)
    attempts = []
    for _ in range(3):
        model = _vehicle_model(
            np.asarray(distances),
            dem,
            int(capacity),
            min(k, n + 2),
            time_windows=np.asarray(windows),
            service=np.asarray(service),
            horizon=float(windows[0][1]),
        )
        status, obj, bound, secs = _solve(model, limit, workers)
        attempts.append({"k_copies": min(k, n + 2), "status": status, "objective": obj, "bound": bound, "seconds": secs})
        if status in ("OPTIMAL", "FEASIBLE"):
            break
        if status == "INFEASIBLE":
            k = min(k * 2, n + 2)
            continue
        break
    best = next((a for a in attempts if a["objective"] is not None), attempts[-1])
    return {**best, "attempts": attempts}


def run_vrptw(seed: int, limit: float, workers: int, jobs: int, out_dir: Path) -> None:
    from llm4ad.task.optimization.vrptw_construct.get_instance import GetData

    instances = GetData(n_instance=16, n_cities=50, seed=seed).generate_instances()
    out_path = out_dir / f"reference_optima_vrptw_seed{seed}.json"
    rows = _load_existing(out_path)
    payload = [
        (k, (instance, limit, workers))
        for k, instance in enumerate(instances)
        if str(k) not in rows or not _cpsat_resumable(rows[str(k)])
    ]
    _run_pool(payload, _vrptw_row, rows, out_path, lambda row: row["status"] == "OPTIMAL", label=f"vrptw_seed{seed}", jobs=jobs)


# ---------------------------------------------------------------------------
# PyVRP(HGS)：CVRP / VRPTW 的近优参考
# ---------------------------------------------------------------------------

def _pyvrp_row(args: tuple[str, object, int, float, int]) -> dict:
    kind, instance, vehicles, seconds, seed = args
    from pyvrp import Model
    from pyvrp.stop import MaxRuntime

    if kind == "cvrp":
        coordinates = instance[:, 1:]
        demands = instance[1:, 0]
        capacity = 50
        service = windows = None
    else:
        coordinates, distances, demands_all, capacity, service, windows = instance
        demands = demands_all[1:]
        service = service[1:]
        windows = windows[1:]
    n = len(demands)

    def xy(i):
        return int(round(float(coordinates[i][0]) * SCALE)), int(round(float(coordinates[i][1]) * SCALE))

    dist = (
        np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
        if kind == "cvrp"
        else np.asarray(distances)
    )
    di = np.round(dist * SCALE).astype(int)

    model = Model()

    def client_kwargs(base: dict, c: int) -> dict:
        if kind == "vrptw":
            base.update(
                service_duration=int(round(float(service[c - 1]) * SCALE)),
                tw_early=int(round(float(windows[c - 1][0]) * SCALE)),
                tw_late=int(round(float(windows[c - 1][1]) * SCALE)),
            )
        return base

    if hasattr(Model, "add_location"):  # pyvrp >= 0.14：先建 location 再挂类型
        endpoints = [model.add_location(*xy(0))]
        for c in range(1, n + 1):
            endpoints.append(model.add_location(*xy(c)))
        model.add_depot(endpoints[0])
        for c in range(1, n + 1):
            model.add_client(**client_kwargs({"location": endpoints[c], "delivery": int(demands[c - 1])}, c))
    else:  # pyvrp 0.11：直接以坐标建 depot/client
        endpoints = [model.add_depot(*xy(0))]
        for c in range(1, n + 1):
            endpoints.append(
                model.add_client(**client_kwargs({"x": xy(c)[0], "y": xy(c)[1], "delivery": int(demands[c - 1])}, c))
            )
    model.add_vehicle_type(num_available=vehicles, capacity=int(capacity))
    for a in range(n + 1):
        for b in range(n + 1):
            if a != b:
                model.add_edge(endpoints[a], endpoints[b], int(di[a][b]), duration=int(di[a][b]))
    result = model.solve(stop=MaxRuntime(seconds), seed=seed, display=False)
    return {
        "status": "HGS_FEASIBLE" if result.is_feasible() else "HGS_INFEASIBLE",
        "objective": result.cost() / SCALE,
        "seconds": seconds,
        "seed": seed,
    }


def run_pyvrp(task: str, split_or_seed: str, seconds: float, jobs: int, out_dir: Path) -> None:  # task: cvrp|vrptw
    if task == "cvrp":
        from llm4ad.task.optimization.cvrp_aco.dataset import load_split_instances

        instances, metadata = load_split_instances(split_or_seed)
        vehicles = len(instances[0]) - 1
        args_list = [("cvrp", instance, vehicles, seconds, 1234 + k) for k, instance in enumerate(instances)]
        label = f"cvrp_{split_or_seed}_hgs"
    else:
        from llm4ad.task.optimization.vrptw_construct.get_instance import GetData

        seed = int(split_or_seed)
        instances = GetData(n_instance=16, n_cities=50, seed=seed).generate_instances()
        args_list = [("vrptw", instance, 51, seconds, 1234 + k) for k, instance in enumerate(instances)]
        label = f"vrptw_seed{seed}_hgs"
    out_path = out_dir / f"reference_optima_{label}.json"
    rows = _load_existing(out_path)
    payload = [
        (k, args)
        for k, args in enumerate(args_list)
        if str(k) not in rows or not _hgs_resumable(rows[str(k)])
    ]
    _run_pool(payload, _pyvrp_row, rows, out_path, lambda row: False, label=label, jobs=jobs)


# ---------------------------------------------------------------------------
# online_bin_packing（本地即可，秒级）
# ---------------------------------------------------------------------------

def _first_fit(sizes, capacity: int) -> int:
    caps: list[int] = []
    for s in sizes:
        for j in range(len(caps)):
            if caps[j] >= s:
                caps[j] -= s
                break
        else:
            caps.append(capacity - s)
    return len(caps)


def _best_fit(sizes, capacity: int) -> int:
    caps: list[int] = []
    for s in sizes:
        j = bisect_left(caps, s)
        if j < len(caps):
            insort(caps, caps.pop(j) - s)
        else:
            insort(caps, capacity - s)
    return len(caps)


def _martello_toth_l2(sizes, capacity: int) -> int:
    arr = np.asarray(sizes)
    best = math.ceil(arr.sum() / capacity)
    for alpha in range(0, capacity // 2 + 1):
        j1 = arr > capacity - alpha
        j2 = (arr > capacity / 2) & (arr <= capacity - alpha)
        j3 = (arr >= alpha) & (arr <= capacity / 2)
        free = int(j2.sum()) * capacity - int(arr[j2].sum())
        need = max(0.0, float(arr[j3].sum()) - free)
        best = max(best, int(j1.sum()) + int(j2.sum()) + math.ceil(need / capacity - 1e-9))
    return int(best)


def run_obp(out_dir: Path) -> None:
    from llm4ad.task.optimization.generated_data_config import get_generated_task_kwargs
    from llm4ad.task.optimization.online_bin_packing.generate_weibull_instances import (
        generate_weibull_multiscale_dataset,
    )

    base = get_generated_task_kwargs("online_bin_packing", "eval")
    data = generate_weibull_multiscale_dataset(base["dataset_specs"], seed=base["seed"])
    out = {}
    for spec in base["dataset_specs"]:
        for capacity in spec["capacities"]:
            size_label = f"{spec['n_items'] // 1000}k" if spec["n_items"] % 1000 == 0 else str(spec["n_items"])
            key = f"{size_label}_{capacity}"
            rows = []
            for k in range(int(spec["n_instances"])):
                items = data[f"{key}_instance_{k}"]["items"]
                lb = _martello_toth_l2(items, capacity)
                ffd = _first_fit(sorted(items, reverse=True), capacity)
                rows.append(
                    {
                        "lower_bound": lb,
                        "ffd": ffd,
                        "first_fit": _first_fit(items, capacity),
                        "best_fit": _best_fit(items, capacity),
                        "ffd_proves_optimum": ffd == lb,
                    }
                )
            out[key] = {
                "instances": rows,
                "mean_lower_bound": float(np.mean([r["lower_bound"] for r in rows])),
                "mean_ffd": float(np.mean([r["ffd"] for r in rows])),
                "mean_first_fit": float(np.mean([r["first_fit"] for r in rows])),
                "mean_best_fit": float(np.mean([r["best_fit"] for r in rows])),
            }
            block = out[key]
            print(
                f"{key}: LB {block['mean_lower_bound']:.2f}  FFD {block['mean_ffd']:.2f}  "
                f"FF {block['mean_first_fit']:.2f}  BF {block['mean_best_fit']:.2f}",
                flush=True,
            )
    payload = {"created_at": datetime.now().isoformat(timespec="seconds"), "task": "obp", "results": out}
    (out_dir / "reference_optima_obp.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# 并行调度与断点续跑
# ---------------------------------------------------------------------------

def _cpsat_resumable(row: dict) -> bool:
    return row.get("status") == "OPTIMAL"


def _hgs_resumable(row: dict) -> bool:
    return True


def _load_existing(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("instances", {})


def _run_pool(payload, worker, rows, out_path, is_final, label, jobs: int) -> None:
    """payload: [(instance_index, args), ...]；imap 保序，按索引写回。"""
    if payload:
        done = 0
        indices = [k for k, _ in payload]
        with get_context("spawn").Pool(processes=min(len(payload), jobs)) as pool:
            for index, row in zip(indices, pool.imap(worker, [args for _, args in payload])):
                rows[str(index)] = row
                done += 1
                status = row.get("status", "?")
                obj = row.get("objective")
                print(
                    f"[{label}] {done}/{len(payload)} status={status}"
                    + (f" objective={obj:.6f}" if isinstance(obj, float) else ""),
                    flush=True,
                )
                _write(out_path, rows, label)
    objectives = [r["objective"] for r in rows.values() if isinstance(r.get("objective"), float)]
    stats = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "num_instances": len(rows),
        "all_optimal": all(is_final(r) for r in rows.values()) if rows else None,
        "mean_objective": float(np.mean(objectives)) if objectives else None,
        "instances": rows,
    }
    out_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[{label}] mean={stats['mean_objective']} all_optimal={stats['all_optimal']} -> {out_path}", flush=True)


def _write(out_path: Path, rows: dict, label: str) -> None:
    out_path.write_text(
        json.dumps({"label": label, "instances": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["tsp", "cvrp", "op", "vrptw", "obp", "pyvrp"], required=True)
    ap.add_argument("--family", choices=["cvrp", "vrptw"], default="cvrp", help="pyvrp 任务族")
    ap.add_argument("--sizes", default="50,100,200", help="tsp sizes")
    ap.add_argument("--seed", type=int, default=2025, help="tsp/vrptw seed")
    ap.add_argument("--split", default="test_50", help="cvrp/op split name")
    ap.add_argument("--solver", choices=["cpsat", "pyvrp"], default="cpsat")
    ap.add_argument("--limit", type=float, default=300.0, help="求解时限（秒/实例；pyvrp 为 runtime）")
    ap.add_argument("--jobs", type=int, default=8, help="并行实例数")
    ap.add_argument("--workers", type=int, default=2, help="每实例 CP-SAT 线程数")
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments" / "_logs")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.task == "tsp":
        run_tsp([int(s) for s in args.sizes.split(",")], args.seed, args.limit, args.workers, args.jobs, args.output_dir)
    elif args.task == "op":
        run_op(args.split, args.limit, args.workers, args.jobs, args.output_dir)
    elif args.task == "cvrp":
        run_cvrp(args.split, args.limit, args.workers, args.jobs, args.output_dir)
    elif args.task == "vrptw":
        run_vrptw(args.seed, args.limit, args.workers, args.jobs, args.output_dir)
    elif args.task == "obp":
        run_obp(args.output_dir)
    elif args.task == "pyvrp":
        run_pyvrp(args.family, str(args.seed) if args.family == "vrptw" else args.split, args.limit, args.jobs, args.output_dir)
    print(f"elapsed {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
