import numpy as np
import networkx as nx

def get_top_accounts_pagerank(graph: nx.Graph) -> list[tuple[str, float]]:
    pagerank_scores = nx.pagerank(graph, alpha=0.85)
    top_accounts = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    return top_accounts


def find_accounts_within_degree(graph: nx.Graph, account_id: str, n: int) -> list[tuple[str, float]]:
    """
    Finds all accounts within degree 1-n of the specified account.
    """
    distances = nx.single_source_shortest_path_length(graph, source=account_id, cutoff=n)
    accounts_within_2 = list(distances.keys())

    return accounts_within_2


def find_nearest_neighbors(features: dict[str, np.ndarray], account_id: str, top_n: int = 5) -> list[tuple[str, float]]:
    if account_id not in features:
        return []

    target = features[account_id]
    target_norm = np.linalg.norm(target)

    scores = []
    for uid, vec in features.items():
        if uid == account_id:
            continue
        if vec.shape != target.shape:
            continue
        denom = target_norm * np.linalg.norm(vec)
        if denom == 0:
            continue
        similarity = float(np.dot(target, vec) / denom)
        scores.append((uid, similarity))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]