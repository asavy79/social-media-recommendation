import numpy as np
import os


def get_edgelist(filename: str) -> list[tuple[int, int]]:
    edges = []
    with open(filename, "r") as f:
        for line in f:
            a, b = line.strip().split()
            edges.append((int(a), int(b)))
    return edges


def _read_featnames(path: str) -> list[str]:
    names = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, name = line.split(" ", 1)
            names.append(name)
    return names


def load_global_features(
    ego_accounts_dir: str,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """
    This function is necessary, as different .feat files for each ego-network have different meanings.
    This takes all of the featnames listed in the .featnames files, and combines them to map each name to an index
    in our global feature vector.

    It will then go through each ego network user, and set the bits in the global feature vector for that user, using the mapping we created.
    """
    ego_ids = sorted(
        {f.split(".")[0] for f in os.listdir(ego_accounts_dir) if f.endswith(".featnames")},
        key=int,
    )

    ego_featnames: dict[str, list[str]] = {}
    vocab_set: set[str] = set()
    for ego in ego_ids:
        names = _read_featnames(os.path.join(ego_accounts_dir, f"{ego}.featnames"))
        ego_featnames[ego] = names
        vocab_set.update(names)

    vocab = sorted(vocab_set)
    name_to_global = {name: i for i, name in enumerate(vocab)}
    dim = len(vocab)

    features: dict[str, np.ndarray] = {}

    def set_bits(user_id: str, local_bits: list[int], local_names: list[str]) -> None:
        vec = features.get(user_id)
        if vec is None:
            vec = np.zeros(dim, dtype=float)
            features[user_id] = vec
        for local_idx, bit in enumerate(local_bits):
            if bit:
                vec[name_to_global[local_names[local_idx]]] = 1.0

    for ego in ego_ids:
        local_names = ego_featnames[ego]

        feat_path = os.path.join(ego_accounts_dir, f"{ego}.feat")
        with open(feat_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                uid = parts[0]
                bits = [int(x) for x in parts[1:]]
                set_bits(uid, bits, local_names)

        egofeat_path = os.path.join(ego_accounts_dir, f"{ego}.egofeat")
        if os.path.exists(egofeat_path):
            with open(egofeat_path) as f:
                bits = [int(x) for x in f.read().split()]
            if bits:
                set_bits(ego, bits, local_names)

    return features, vocab
