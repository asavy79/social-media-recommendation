# SNAP Facebook Dataset: Structural & Feature Layout

This document provides a technical breakdown of the Stanford SNAP ego-Facebook dataset. It explains how to interpret the graph structure, anonymized metadata, and social groupings for building recommendation engines.

---

## 1. The Global Graph (`facebook_combined.txt`)

This is the primary file for building a recommendation engine. It merges all 10 ego-networks into a single **undirected graph**.

- **Format:** `NodeID_1 NodeID_2` (Space-separated).
- **Nodes:** 4,039 (Anonymized integers from 0 to 4038).
- **Edges:** 88,234.
- **Logic:** Each line represents a friendship. If `0 1` exists, User 0 and User 1 are friends.
- **Usage:** This is the best file to load into `NetworkX`. It includes the "0" node and all other hubs that are often missing from the individual `.edges` files.

---

## 2. Ego-Network Structure (`[node_id].edges`)

These files represent the local "worldview" of the 10 central users (the "Egos") who shared their data.

- **The Missing Hub:** The central Ego node (the filename ID) is **omitted** from its own `.edges` file because it is connected to every node mentioned in that file by default.
- **Content:** These lines show the connections _between_ the friends of the Ego.
- **Example:** In `0.edges`, if you see `1 5`, it means Friend 1 and Friend 5 are friends with each other (a "mutual friend" connection for User 0).

---

## 3. Metadata & Features (The Profiles)

The metadata is "anonymized," meaning specific names are replaced with IDs, but the mathematical relationships remain intact.

### A. Feature Names (`[node_id].featnames`)

This acts as the **Data Dictionary**. It defines what each bit in the binary vector represents.

- **Format:** `[Column_Index] [Category];[Sub-category];[Anonymized_Feature_ID]`
- **Example:** `3 education;school;id;anonymized_feature_50`
  - _Interpretation:_ The 4th column (index 3) in the `.feat` file represents a specific school.

### B. Feature Vectors (`[node_id].feat`)

This file contains the "Profile" for every friend in that ego-network.

- **Format:** `[NodeID] [Bit_0] [Bit_1] [Bit_2] ... [Bit_N]`
- **Binary Logic:** A `1` means the user has that attribute; `0` means they don't.
- **Usage:** Used for **Content-Based Recommendations**. You can calculate similarity by comparing these bitstrings.

### C. Ego Features (`[node_id].egofeat`)

- **Format:** A single row of binary data.
- **Purpose:** This is the feature vector (profile) for the central Ego-user themselves.

---

## 4. Manual Social Groupings (`[node_id].circles`)

These are "Ground Truth" communities where users manually categorized their friends.

- **Format:** `circle[Number] [NodeID_1] [NodeID_2] ...`
- **Context:** While edges show _who knows who_, circles show _why_ they know each other (e.g., "High School Friends," "Work Colleagues").
- **Usage:** Use this to test if your recommendation algorithm correctly identifies "groups" or "cliques."

---

## 5. Summary Table for Implementation

| Task                                   | File(s) to Use                               |
| :------------------------------------- | :------------------------------------------- |
| **Calculate Popularity (PageRank)**    | `facebook_combined.txt`                      |
| **Find Friends of Friends (Degree-2)** | `facebook_combined.txt`                      |
| **Recommend by Shared Interest**       | `[node].featnames` + `[node].feat`           |
| **Verify Community Logic**             | `[node].circles`                             |
| **Building a Global Profile Map**      | Combine all `[node].feat` + `[node].egofeat` |

---

**Data Source:** [Stanford Network Analysis Project (SNAP)](https://snap.stanford.edu/data/ego-Facebook.html
