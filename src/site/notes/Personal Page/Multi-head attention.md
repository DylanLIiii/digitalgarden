---
{"dg-publish":true,"permalink":"/personal-page/multi-head-attention/"}
---

### review 
- [[Personal Page/Transformer\|Transformer]]
- [[Personal Page/self-attention mechanism\|self-attention mechanism]]


Multi-head attention is a mechanism used in transformer-based models to perform self-attention on multiple different representations or subspaces of the input data simultaneously. This allows the model to attend to different parts of the input data and combine the information in a more effective way.

To understand how multi-head attention works, it's important to first understand the concept of self-attention. Self-attention allows a model to attend to different parts of the input data at the same time, rather than processing the data sequentially. This is done by using a dot product between the input data and a set of learnable weights, called "attention weights". The attention weights are used to compute a weighted sum of the input data, which is then used to generate the output of the self-attention layer.

In multi-head attention, the self-attention mechanism is applied multiple times, with each attention head operating on a different representation or subspace of the input data. This allows the model to attend to different aspects of the input data simultaneously, which can be beneficial for tasks where the relationships between different parts of the input data are complex and cannot be captured by a single attention head.

Mathematically, the multi-head attention mechanism can be represented as follows:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, ..., \text{head}_h)W^O$$

where $Q$, $K$, and $V$ are the query, key, and value matrices, respectively. The attention heads are computed as:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

where $W_i^Q$, $W_i^K$, and $W_i^V$ are the learnable weights for the $i$-th attention head. The attention function is typically the scaled dot-product attention, which is defined as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $d_k$ is the dimension of the keys.


```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, input_size, output_size):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(input_size, n_heads, output_size)
        
    def forward(self, input):
        # Split the input into multiple heads
        input_heads = torch.split(input, split_size_or_sections=input.size(0), dim=0)
        
        # Apply attention to each head
        attention_outputs = []
        for input_head in input_heads:
            attention_output, _ = self.attention(input_head, input_head, input_head)
            attention_outputs.append(attention_output)
        
        # Concatenate the attention outputs
        output = torch.cat(attention_outputs, dim=0)
        return output

attention_layer = MultiHeadAttention(n_heads=8, input_size=64, output_size=64)
input_data = torch.randn(10, 32, 64)
output = attention_layer(input_data)

```