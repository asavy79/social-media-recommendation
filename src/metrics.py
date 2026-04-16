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


def find_nearest_neighbors(account_id: str) -> None:
    """
    This will use the cosine similarity of the feature vectors to find the nearest neigbhors.
    Look in data/ego_accounts files that end in .featnames and .feat to fully understand the data
    """
    pass