import networkx as nx
from parser import load_features
from metrics import get_top_accounts_pagerank, find_accounts_within_degree, find_nearest_neighbors

GRAPH_FILE = "data/facebook_combined.txt"
EGO_ACCOUNTS_DIR = "data/ego_accounts"


if __name__ == "__main__":
    G = nx.read_edgelist(GRAPH_FILE, create_using=nx.Graph())
    features = load_features(EGO_ACCOUNTS_DIR)

    print("Top influencers by PageRank:")
    print(get_top_accounts_pagerank(G))

    print("\nAccounts within 2 degrees of user 107:")
    print(find_accounts_within_degree(G, "107", 2))

    print("\nNearest neighbors to user 107 by profile similarity:")
    print(find_nearest_neighbors(features, "107"))
