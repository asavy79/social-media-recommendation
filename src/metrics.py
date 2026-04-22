import numpy as np
import networkx as nx

def get_top_accounts_pagerank(graph: nx.Graph, account_id: str = None) -> list[tuple[str, float]]:
    # use personalized pagerank if an account id is provided
    if account_id is not None:
        personalization = {account_id: 1}
    else:
        personalization = None
    pagerank_scores = nx.pagerank(graph, alpha=0.85, personalization=personalization)
    top_accounts = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    return top_accounts


def find_accounts_within_degree(graph: nx.Graph, account_id: str, n: int) -> list[tuple[str, float]]:
    """
    Finds all accounts within degree 1-n of the specified account.
    """
    distances = nx.single_source_shortest_path_length(graph, source=account_id, cutoff=n)
    accounts_within_2 = list(distances.keys())

    return accounts_within_2


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def find_nearest_neighbors(features: dict[str, np.ndarray], account_id: str, top_n: int = 5) -> list[tuple[str, float]]:
    if account_id not in features:
        return []

    target = features[account_id]
    if np.linalg.norm(target) == 0:
        return []

    scores = []
    for uid, vec in features.items():
        if uid == account_id:
            continue
        if vec.shape != target.shape or np.linalg.norm(vec) == 0:
            continue
        scores.append((uid, _cosine(target, vec)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


def recommend_accounts(
    graph: nx.Graph,
    features: dict[str, np.ndarray],
    account_id: str,
    top_n: int = 10,
    max_hops: int = 3,
    alpha: float = 0.5,
) -> list[tuple[str, float, float, float]]:
    """
    This recommends accounts for a certain account by combining personalized PageRank and cosine similarity.
    Increasing the alpha increases the weight of the PageRank scores while decrerasing it increases the importance of cosine similarity.
    You can also adjust the max_hops to the number of degrees of separation you want to consider.

    We will return tuples of (account_id, combined_score, raw_ppr, raw_cosine)
    """
    if account_id not in graph:
        return []

    ppr = nx.pagerank(graph, alpha=0.85, personalization={account_id: 1.0})

    reachable = nx.single_source_shortest_path_length(graph, source=account_id, cutoff=max_hops)

    # We will exclude the neighbors, as they are already friends of the ego account.
    excluded = set(graph.neighbors(account_id)) | {account_id}
    candidates = [n for n in reachable if n not in excluded]
    if not candidates:
        return []

    ppr_vals = np.array([ppr.get(n, 0.0) for n in candidates])

    target_vec = features.get(account_id)
    if target_vec is None:
        cos_vals = np.zeros(len(candidates))
    else:
        cos_vals = np.array([
            _cosine(target_vec, features[n]) if n in features else 0.0
            for n in candidates
        ])

    # We normalize the PageRank and cosine similarity scores, as their scales are very different.
    ppr_max = ppr_vals.max()
    ppr_norm = ppr_vals / ppr_max if ppr_max > 0 else ppr_vals
    cos_norm = (cos_vals + 1.0) / 2.0

    combined = alpha * ppr_norm + (1.0 - alpha) * cos_norm

    results = [
        (candidates[i], float(combined[i]), float(ppr_vals[i]), float(cos_vals[i]))
        for i in range(len(candidates))
    ]
    results.sort(key=lambda r: r[1], reverse=True)
    return results[:top_n]