import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from parser import load_global_features


def test_same_features_across_egos_produce_equal_global_vectors(tmp_path):
    """
    Two egos with the same featname strings but in different local orderings.
    User "A" lives in ego 1 and has birthday+school set.
    User "B" lives in ego 2 and has the same two features set.
    After global re-encoding, A and B must have identical feature vectors.
    """
    (tmp_path / "1.featnames").write_text(
        "0 birthday;anonymized feature 5\n"
        "1 education;school;anonymized feature 10\n"
    )
    (tmp_path / "1.feat").write_text("A 1 1\n")
    (tmp_path / "1.egofeat").write_text("0 0\n")

    (tmp_path / "2.featnames").write_text(
        "0 education;school;anonymized feature 10\n"
        "1 birthday;anonymized feature 5\n"
    )
    (tmp_path / "2.feat").write_text("B 1 1\n")
    (tmp_path / "2.egofeat").write_text("0 0\n")

    features, _ = load_global_features(str(tmp_path))

    assert (features["A"] == features["B"]).all()
