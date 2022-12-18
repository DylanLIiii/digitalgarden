---
{"dg-publish":true,"permalink":"/personal-page/transformer/"}
---

The transformer consists of multiple layers of processing units, each of which consists of a self-attention mechanism and a feedforward network. The self-attention mechanism allows the model to weight the input features and model their interactions, while the feedforward network allows the model to transform the weighted input features into a new representation.

The self-attention mechanism in the transformer is based on the dot product between query and key vectors, which is used to measure the similarity between the two. Given a set of input features ${\textbf{x}_1, \textbf{x}_2, ..., \textbf{x}_N}$, the self-attention mechanism calculates the output at each position as a weighted sum of the input features, where the weights are calculated using the following equation:

$$\textbf{y}_i = \sum_{j=1}^{N} \frac{\exp(\textbf{q}_i^T \textbf{k}_j)}{\sum_{k=1}^{N} \exp(\textbf{q}_i^T \textbf{k}_k)}\textbf{v}_j$$

where $\textbf{q}_i$, $\textbf{k}_j$, and $\textbf{v}_j$ are the query, key, and value vectors for the input feature at position $i$, respectively. The query and key vectors are typically obtained by projecting the input feature onto a lower-dimensional space using linear transformations:

$$\textbf{q}_i = \textbf{W}_q \textbf{x}_i$$

$$\textbf{k}_i = \textbf{W}_k \textbf{x}_i$$

$$\textbf{v}_i = \textbf{W}_v \textbf{x}_i$$

where $\textbf{W}_q$, $\textbf{W}_k$, and $\textbf{W}_v$ are learnable weight matrices.

The dot product between the query and key vectors is used to measure the similarity between the two, and the exponential function is used to weight the input features based on this similarity. The value vector is then used to transform the weighted input feature into a new representation.

The output of the self-attention mechanism is then passed through a feedforward network, which consists of one or more fully-connected layers. This allows the transformer to transform the weighted input features into a new representation that is more suited to the task at hand.
