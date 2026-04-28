import os
import random

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

from parser import load_global_features, load_graph, hide_user_edges
from evaluation import evaluate_users
from metrics import recommend_accounts

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EGO_DIR = os.path.join(DATA_DIR, "ego_accounts")
COMBINED_EDGES = os.path.join(DATA_DIR, "facebook_combined.txt")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")

TOP_N_LIST = [5, 10, 20]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
P_HIDE = 0.2
SEED = 42
TARGET_USER = "107"


def plot_precision_recall(rows: list[dict], top_n_list: list[int]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Recommendation Performance vs Alpha\n(alpha=1 → PageRank only, alpha=0 → Cosine only)")

    alphas = [r["alpha"] for r in rows]

    for n in top_n_list:
        ax1.plot(alphas, [r[f"P@{n}"] for r in rows], marker="o", label=f"P@{n}")
        ax2.plot(alphas, [r[f"R@{n}"] for r in rows], marker="o", label=f"R@{n}")

    ax1.set_title("Precision")
    ax1.set_xlabel("Alpha (PageRank weight)")
    ax1.set_ylabel("Score")
    ax1.legend()
    ax1.set_xticks(alphas)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.set_title("Recall")
    ax2.set_xlabel("Alpha (PageRank weight)")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.set_xticks(alphas)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Shows how precision and recall change as alpha shifts the balance between
    # PageRank (network position) and cosine similarity (shared profile features).
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "precision_recall.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


def plot_network(G: nx.Graph, features: dict, user: str, alpha: float = 0.5, top_n: int = 10, filename: str = "network_graph.png") -> None:
    rng = random.Random(SEED)
    G_train, hidden_edges = hide_user_edges(G, user, p=P_HIDE, rng=rng)
    hidden_friends = {v for (_, v) in hidden_edges}

    recs = recommend_accounts(G_train, features, user, top_n=top_n, alpha=alpha)
    rec_ids = {r[0] for r in recs}

    direct_friends = set(G.neighbors(user))

    nodes_to_show = {user} | direct_friends | rec_ids
    subgraph = G.subgraph(nodes_to_show)

    color_map = []
    for node in subgraph.nodes():
        if node == user:
            color_map.append("#e74c3c")       # red — target user
        elif node in hidden_friends and node in rec_ids:
            color_map.append("#2ecc71")       # green — correctly recommended hidden friend
        elif node in hidden_friends:
            color_map.append("#e67e22")       # orange — hidden friend not recommended
        elif node in rec_ids:
            color_map.append("#9b59b6")       # purple — recommended (not a hidden friend)
        else:
            color_map.append("#95a5a6")       # grey — visible friend

    pos = nx.spring_layout(subgraph, seed=SEED, k=0.4)

    plt.figure(figsize=(12, 9))
    nx.draw_networkx(
        subgraph,
        pos=pos,
        node_color=color_map,
        node_size=120,
        with_labels=False,
        edge_color="#cccccc",
        width=0.5,
    )

    nx.draw_networkx_nodes(subgraph, pos, nodelist=[user], node_color="#e74c3c", node_size=400)

    legend = [
        mpatches.Patch(color="#e74c3c", label=f"Target user ({user})"),
        mpatches.Patch(color="#2ecc71", label="Hidden friend — correctly recommended"),
        mpatches.Patch(color="#e67e22", label="Hidden friend — missed"),
        mpatches.Patch(color="#9b59b6", label="Recommended (not a hidden friend)"),
        mpatches.Patch(color="#95a5a6", label="Visible friend"),
    ]
    plt.legend(handles=legend, loc="upper left", fontsize=8)
    plt.title(f"Friend Recommendation Network for User {user}\n(alpha={alpha}, top-{top_n} recommendations)")
    # Visualizes the friend network around the target user. Hidden friends are colored
    # green if correctly recommended, orange if missed. Purple nodes are false positives.
    plt.axis("off")
    plt.tight_layout()

    out = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading graph and features...")
    G = load_graph(COMBINED_EDGES)
    features, _ = load_global_features(EGO_DIR)

    ego_users = sorted(
        {f.split(".")[0] for f in os.listdir(EGO_DIR) if f.endswith(".featnames")},
        key=int,
    )

    print("Running evaluation for precision/recall chart...")
    rows = evaluate_users(G, features, ego_users, P_HIDE, TOP_N_LIST, ALPHAS, SEED)
    plot_precision_recall(rows, TOP_N_LIST)

    print(f"Building network graph for user {TARGET_USER} (alpha=0.5, top-10)...")
    plot_network(G, features, TARGET_USER, alpha=0.5, top_n=10, filename="network_graph_alpha05.png")

    print(f"Building network graph for user {TARGET_USER} (alpha=1.0, top-20)...")
    plot_network(G, features, TARGET_USER, alpha=1.0, top_n=20, filename="network_graph_alpha10.png")

    print("Done.")
