import networkx as nx

FILENAME = "data/facebook_combined.txt"


if __name__ == "__main__":

    G = nx.read_edgelist(FILENAME, create_using=nx.Graph())

    pagerank_scores = nx.pagerank(G, alpha=0.85)

    top_influencers = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print(top_influencers)