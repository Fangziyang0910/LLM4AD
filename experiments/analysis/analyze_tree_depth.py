"""测量 TraceAAD 树版本（V8 / V8.2）正式批次的轨迹深度。

轨迹深度决定生成上下文中实际可用的历史长度。V8 与 V8.2 的上下文协议
都允许最多 8 条祖先边，但只有当扩展点足够深时该容量才会被填满。

从 `artifacts/edges.jsonl` 的 parent/child 关系重建单父代树，统计：

- 扩展点深度：每次扩展所选父节点到根子节点的路径长度，即该次生成
  实际可获得的祖先历史边数上界；
- 深度 >= 4 的扩展占比：上下文历史超过 3 条边的样本比例；
- 树的最大深度。

用法：
    uv run python experiments/analysis/analyze_tree_depth.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs" / "analysis" / "version_diagnosis"
GLOB = "experiments/*/traceaad_v8/version8/*/artifacts/edges.jsonl"

DEEP_THRESHOLD = 4


def protocol_of(run_name: str) -> str:
    return "V8" if run_name.startswith("v8_") else "V8.2"


def measure(path: Path) -> dict:
    parent_of: dict[int, int] = {}
    pairs: list[tuple[int, int]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            e = json.loads(line)
            pairs.append((e["parent_id"], e["child_id"]))
    for par, ch in pairs:
        parent_of[ch] = par

    cache: dict[int, int] = {}

    def depth(node: int) -> int:
        chain = []
        cur = node
        while cur not in cache and cur in parent_of:
            if cur in chain:  # 环保护，正常单父代树不会触发
                break
            chain.append(cur)
            cur = parent_of[cur]
        base = cache.get(cur, 1)
        for i, n in enumerate(reversed(chain)):
            base += 1
            cache[n] = base
        cache.setdefault(node, base)
        return cache[node]

    exp = np.array([depth(par) for par, _ in pairs])
    all_d = np.array([depth(ch) for _, ch in pairs])
    return {
        "edges": len(pairs),
        "expansion_depth_median": float(np.median(exp)),
        "expansion_depth_mean": float(exp.mean()),
        "expansion_depth_p90": float(np.percentile(exp, 90)),
        "frac_depth_ge_4": float((exp >= DEEP_THRESHOLD).mean()),
        "max_depth": int(all_d.max()),
    }


def main() -> None:
    runs = sorted(REPO.glob(GLOB))
    if not runs:
        print(f"未找到工件: {GLOB}")
        return

    records = []
    for p in runs:
        run = p.parts[-3]
        task = p.relative_to(REPO).parts[1]
        rec = {"run": run, "task": task, "protocol": protocol_of(run), **measure(p)}
        records.append(rec)

    print(f"{len(records)} 个 run，深度阈值 = {DEEP_THRESHOLD}\n")
    header = (f"{'协议':<6}{'任务':<20}{'边数':>6}{'扩展点深度中位':>15}"
              f"{'均值':>8}{'p90':>7}{'深度>=4':>9}{'最大深度':>9}")
    print(header)
    for r in sorted(records, key=lambda r: (r["protocol"], r["task"], r["run"])):
        print(f"{r['protocol']:<6}{r['task']:<20}{r['edges']:>6}"
              f"{r['expansion_depth_median']:>15.0f}{r['expansion_depth_mean']:>8.2f}"
              f"{r['expansion_depth_p90']:>7.0f}{r['frac_depth_ge_4']:>9.1%}"
              f"{r['max_depth']:>9}")

    print("\n=== 按协议 × 任务聚合 ===")
    print(f"{'协议':<6}{'任务':<20}{'扩展点深度中位':>15}{'深度>=4':>9}{'最大深度':>9}")
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["protocol"], r["task"])].append(r)
    aggregate = {}
    for (proto, task), lst in sorted(grouped.items()):
        med = float(np.mean([r["expansion_depth_median"] for r in lst]))
        deep = float(np.mean([r["frac_depth_ge_4"] for r in lst]))
        mx = float(np.mean([r["max_depth"] for r in lst]))
        aggregate[f"{proto}/{task}"] = {
            "expansion_depth_median": med, "frac_depth_ge_4": deep, "max_depth": mx
        }
        print(f"{proto:<6}{task:<20}{med:>15.1f}{deep:>9.1%}{mx:>9.1f}")

    print("\n=== 按协议聚合 ===")
    overall = {}
    for proto in ("V8", "V8.2"):
        lst = [r for r in records if r["protocol"] == proto]
        if not lst:
            continue
        overall[proto] = {
            "expansion_depth_median": float(np.mean([r["expansion_depth_median"] for r in lst])),
            "expansion_depth_mean": float(np.mean([r["expansion_depth_mean"] for r in lst])),
            "frac_depth_ge_4": float(np.mean([r["frac_depth_ge_4"] for r in lst])),
            "max_depth": float(np.mean([r["max_depth"] for r in lst])),
        }
        o = overall[proto]
        print(f"{proto}: 扩展点深度中位 {o['expansion_depth_median']:.2f}, "
              f"均值 {o['expansion_depth_mean']:.2f}, "
              f"深度>=4 占比 {o['frac_depth_ge_4']:.1%}, "
              f"最大深度 {o['max_depth']:.1f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "tree_depth.json"
    out.write_text(
        json.dumps({"deep_threshold": DEEP_THRESHOLD, "runs": records,
                    "by_protocol_task": aggregate, "by_protocol": overall},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n已写出 {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
