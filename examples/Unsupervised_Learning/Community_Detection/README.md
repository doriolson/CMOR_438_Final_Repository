## Community Detection

This directory contains example code and notes for **Community Detection**, an unsupervised learning technique used to identify groups of related nodes within a network or graph. Community detection algorithms aim to discover clusters of nodes that are more densely connected to each other than to the rest of the network. This is widely used in social networks, biological networks, and other relational data.

---

## Algorithm

Community detection treats the dataset as a **graph**, where nodes represent entities (e.g., individuals, items) and edges represent relationships or similarity between them. The goal is to partition the graph into communities such that nodes within the same community are strongly connected, while connections between communities are weaker.

Common approaches include:

- **Modularity-based methods**: These methods optimize a modularity score, which measures the density of edges inside communities compared to what would be expected in a random graph. Higher modularity indicates stronger community structure.  
- **Label propagation**: Nodes iteratively adopt the most frequent label among their neighbors until communities stabilize.  
- **Spectral clustering**: Uses eigenvalues of the graph Laplacian to partition nodes into communities.  

Community detection algorithms are particularly useful for exploring hidden structure in networks, identifying clusters of similar entities, and analyzing relationships that are not immediately obvious from raw data.

**Pros**
- Can uncover hidden structures in relational data  
- Does not require predefined labels or number of communities  
- Flexible, with multiple algorithms suited to different types of networks  

**Cons**
- Results can vary depending on the method and graph representation  
- Sensitive to noisy or sparse data  
- Computationally intensive for very large networks  

---

## Data

For this project, we use the **Obesity dataset** to construct a network based on feature similarity for community detection.  

The dataset is first cleaned and categorical features are encoded using **one-hot encoding**. All features are then **standardized** using `StandardScaler` to ensure that distances used in graph construction are meaningful. The resulting scaled feature matrix serves as the basis for creating a similarity graph, where nodes represent individuals and edges represent proximity or similarity in the feature space.  

After constructing the graph, community detection algorithms are applied to identify clusters of individuals with similar health and lifestyle characteristics. These communities can then be interpreted to explore patterns and relationships within the dataset.
