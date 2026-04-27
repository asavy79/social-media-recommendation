from __future__ import annotations

import random
from typing import Iterable

import networkx as nx
import numpy as np

from metrics import recommend_accounts
from parser import hide_user_edges


def evaluate_user(
    graph: nx.Graph,
    features: dict[str, np.ndarray],
    user: str,
    p_hide: float,
    top_n_list: list[int],
    alpha: float,
    rng: random.Random,
) -> dict[int, tuple[int, int]]:
    G_train, hidden = hide_user_edges(graph, user, p=p_hide, rng=rng)

    hidden_friends = {v for (_, v) in hidden}

    if not hidden_friends:
        return {n: (0, 0) for n in top_n_list}

    max_n = max(top_n_list)
    recs = recommend_accounts(G_train, features, user, top_n=max_n, alpha=alpha)
    rec_ids = [r[0] for r in recs]

    out: dict[int, tuple[int, int]] = {}
    for n in top_n_list:
        hits = len(set(rec_ids[:n]) & hidden_friends)
        out[n] = (hits, len(hidden_friends))
    return out


def evaluate_users(
    graph: nx.Graph,
    features: dict[str, np.ndarray],
    users: Iterable[str],
    p_hide: float,
    top_n_list: list[int],
    alphas: list[float],
    seed: int = 42,
) -> list[dict]:
    user_list = list(users)
    rows: list[dict] = []

    for alpha in alphas:
        per_n: dict[int, dict[str, list[float]]] = {
            n: {"precisions": [], "recalls": []} for n in top_n_list
        }

        for user in user_list:
            user_rng = random.Random(seed + int(user))
            results = evaluate_user(
                graph, features, user, p_hide, top_n_list, alpha, user_rng
            )
            for n, (hits, num_hidden) in results.items():
                if num_hidden == 0:
                    continue
                per_n[n]["precisions"].append(hits / n)
                per_n[n]["recalls"].append(hits / num_hidden)

        row: dict = {"alpha": alpha}
        for n in top_n_list:
            ps = per_n[n]["precisions"]
            rs = per_n[n]["recalls"]
            row[f"P@{n}"] = float(np.mean(ps)) if ps else 0.0
            row[f"R@{n}"] = float(np.mean(rs)) if rs else 0.0
        rows.append(row)

    return rows


def format_results_table(rows: list[dict], top_n_list: list[int]) -> str:
    headers = ["alpha"] + [f"P@{n}" for n in top_n_list] + [f"R@{n}" for n in top_n_list]

    col_width = 8
    header_line = "  ".join(f"{h:>{col_width}}" for h in headers)
    sep = "-" * len(header_line)

    lines = [header_line, sep]
    for row in rows:
        cells = [f"{row['alpha']:>{col_width}.2f}"]
        for n in top_n_list:
            cells.append(f"{row[f'P@{n}']:>{col_width}.3f}")
        for n in top_n_list:
            cells.append(f"{row[f'R@{n}']:>{col_width}.3f}")
        lines.append("  ".join(cells))

    return "\n".join(lines)
