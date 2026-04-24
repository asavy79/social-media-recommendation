import networkx as nx
from parser import load_global_features
from metrics import recommend_accounts

GRAPH_FILE = "data/facebook_combined.txt"
EGO_ACCOUNTS_DIR = "data/ego_accounts"

ACCOUNT_IDS = ["0", "107", "348", "414", "686", "698", "1684", "1912", "3437", "3980"]


#Code verifying that user has the correct data; just incase it's not working properly
def main() -> int:
    graph_path, ego_dir = find_data_paths()

    if not Path(graph_path).exists():
        print(f"Required graph file not found: {graph_path}")
        return 2

    if not Path(ego_dir).exists():
        print(f"Required ego accounts directory not found: {ego_dir}")
        return 2

    G = nx.read_edgelist(graph_path, create_using=nx.Graph())
    features, vocab = load_global_features(ego_dir)

    target = input(f"Enter a target account ID ({', '.join(ACCOUNT_IDS)}): ")
    if target not in ACCOUNT_IDS:
        print(f"Invalid account ID: {target}")
        exit(1)

    print(f"Global feature vocab size: {len(vocab)}")
    print(f"Users with features: {len(features)} (graph has {G.number_of_nodes()} nodes)")

    print(f"\nTop account recommendations for user {target}:")
    print(f"  {'id':>6}  {'score':>8}  {'ppr':>10}  {'cosine':>8}")
    recommended = recommend_accounts(G, features, target, top_n=10)
    for uid, score, ppr, cosine in recommended:
        print(f"  {uid:>6}  {score:>8.4f}  {ppr:>10.6f}  {cosine:>8.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

