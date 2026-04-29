# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import pandas as pd
from tabulate import tabulate

def visualize_graph(G, recommended, target):
    nodes = [t[0] for t in recommended]
    
    H = G.subgraph(nodes).copy()
   

    pos = nx.spring_layout(H, seed=42,  k=1.5)
    edge_labels = nx.get_edge_attributes(H, "score")

 

    nx.draw(H, pos, with_labels=True, node_size=2000)
    nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels)

    plt.show()


def show_most_common_traits(features, vocab, recommended, target,n):
    feature_list= []
    for t in recommended:
        feature_list.append(np.array(features[t[0]], dtype=int))
    
    running = []
    acc = 0
    target_vector = features[target]
    #print(feature_list[0])
    for f in feature_list:
        for i in range(len(f)):
            if(f[i]==1) and (target_vector[i]==1):
                running.append(i)
    common_features_len = len(running)
    if(n> common_features_len):
        n = common_features_len
    top3 = Counter(running).most_common(n)
    df = pd.DataFrame({'Feature Name': [], 'Count': []})
    
    #agg_df = pd.concat([df, agg_df], ignore_index=True)
    for item, count in top3:
        #print(f"Trait {vocab[item]} was shared in common with other recommended users {count} times")
        df.loc[len(df)] = [vocab[item], count]
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    return 1
    
    
    
    

# %%
