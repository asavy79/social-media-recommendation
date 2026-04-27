import os

from parser import load_global_features, load_graph
from evaluation import evaluate_users, format_results_table


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EGO_DIR = os.path.join(DATA_DIR, "ego_accounts")
COMBINED_EDGES = os.path.join(DATA_DIR, "facebook_combined.txt")


if __name__ == "__main__":
    G = load_graph(COMBINED_EDGES)
    features, _ = load_global_features(EGO_DIR)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    ego_users = sorted(
        {f.split(".")[0] for f in os.listdir(EGO_DIR) if f.endswith(".featnames")},
        key=int,
    )
    print(f"Evaluating on {len(ego_users)} ego users: {ego_users}")

    p_hide = 0.2
    top_n_list = [5, 10, 20]
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    seed = 42

    rows = evaluate_users(
        graph=G,
        features=features,
        users=ego_users,
        p_hide=p_hide,
        top_n_list=top_n_list,
        alphas=alphas,
        seed=seed,
    )

    print(
        f"\nResults (p_hide={p_hide}, seed={seed}, n_users={len(ego_users)}, "
        f"alpha=PPR weight, 1-alpha=cosine weight)\n"
    )
    print(format_results_table(rows, top_n_list))
