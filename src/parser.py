import numpy as np
import os


def get_edgelist(filename: str) -> list[tuple[int, int]]:
    edges = []
    with open(filename, "r") as f:
        for line in f:
            a, b = line.strip().split()
            edges.append((int(a), int(b)))
    return edges


def load_features(ego_accounts_dir: str) -> dict[str, np.ndarray]:
    features = {}
    for filename in os.listdir(ego_accounts_dir):
        if not filename.endswith(".feat"):
            continue
        with open(os.path.join(ego_accounts_dir, filename)) as f:
            for line in f:
                parts = line.strip().split()
                features[parts[0]] = np.array(parts[1:], dtype=float)
    return features
