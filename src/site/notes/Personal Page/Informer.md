---
{"dg-publish":true,"permalink":"/personal-page/informer/"}
---


To begin, let's first provide some background on transformer models and self-attention mechanisms.

- [[Personal Page/Transformer\|Transformer]]
- [[Personal Page/self-attention mechanism\|self-attention mechanism]]
- [[Personal Page/Multi-head attention\|Multi-head attention]]

One issue with transformer models is that they can be computationally expensive, especially when processing long sequences of text. The Informer model addresses this issue by introducing several modifications to the transformer architecture that allow it to handle long sequences more efficiently.

## Informer: Beyond Efficient Transformer

As mentioned previously, the Informer model was introduced in the paper "Informer: Beyond Efficient Transformer for Long Text Generation" by Liu et al. and aims to improve the efficiency and effectiveness of transformer models for long text generation tasks.

One issue with transformer models is that they can be computationally expensive, especially when processing long sequences of text. The Informer model addresses this issue by introducing several modifications to the transformer architecture that allow it to handle long sequences more efficiently.

### Informer architecture

The Informer model is based on the transformer architecture, but includes several modifications to improve efficiency. These modifications include:

-   **Adaptive input encoding**: The Informer model uses an adaptive input encoding scheme that reduces the number of input tokens that need to be processed by the transformer model. This is achieved by replacing consecutive tokens that have the same token type with a single token that represents the entire sequence. For example, a sequence "I am a student" might be encoded as "I am_a student" where the "_a" token represents the sequence "a student".
    
-   **Adaptive attention**: The Informer model also introduces an adaptive attention mechanism that reduces the number of attention calculations that need to be performed by the model. This is achieved by using a combination of global and local attention mechanisms, where global attention is used for long-range dependencies and local attention is used for shorter-range dependencies.
    
-   **Adaptive positional encoding**: The Informer model uses an adaptive positional encoding scheme that reduces the number of positional embeddings that need to be calculated. This is achieved by using a combination of fixed and learnable positional embeddings, where the fixed positional embeddings are used for common positions in the input sequence and the learnable positional embeddings are used for less common positions.
    

### Informer loss function

The Informer model uses a modified version of the cross-entropy loss function for training. This loss function is defined as:

$$Loss = -\sum_{i=1}^{n} y_i log(\hat{y_i})$$

where $y_i$ is the true label for the $i$-th token in the sequence and $\hat{y_i}$ is the predicted label.

In addition to the cross-entropy loss, the Informer model also includes a coverage loss term that penalizes the model for repeating the same attention weights over multiple time steps. This loss term is defined as:

$$Loss_{coverage} = \sum_{t=1}^{T} \sum_{i=1}^{n} min(c_{t,i}, \hat{c}_{t,i})$$

where $c_{t,i}$ and $\hat{c}_{t,i}$ are the true and predicted attention weights for the $i$-th token at time step $t$, and $T$ is the number of time steps.

The total loss for the Informer model is then the sum of the cross-entropy loss and the coverage loss:

$$Loss_{total} = Loss + \lambda Loss_{coverage}$$

where $\lambda$ is a hyperparameter that controls the weight of the coverage loss term.

## Realizing the Informer model 
```python
import torch
import torch.nn as nn

class Informer(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_heads, num_classes):
    super(Informer, self).__init__()
    
    # Adaptive input encoding
    self.input_encoder = nn.Linear(input_size, hidden_size)
    
    # Adaptive attention
    self.global_attention = nn.Linear(hidden_size, hidden_size)
    self.local_attention = nn.Linear(hidden_size, hidden_size)
    
    # Adaptive positional encoding
    self.fixed_positional_embeddings = nn.Embedding(num_positions, hidden_size)
    self.learnable_positional_embeddings = nn.Linear(num_positions, hidden_size)
    
    # Transformer layers
    self.transformer_layers = nn.ModuleList([
      TransformerLayer(hidden_size, num_heads)
      for _ in range(num_layers)
    ])
    
    # Output layer
    self.output_layer = nn.Linear(hidden_size, num_classes)
    
  def forward(self, inputs):
    # Adaptive input encoding
    encoded_inputs = self.input_encoder(inputs)
    
    # Adaptive attention
    global_attention = self.global_attention(encoded_inputs)
    local_attention = self.local_attention(encoded_inputs)
    
    # Adaptive positional encoding
    fixed_positional_embeddings = self.fixed_positional_embeddings(positions)
    learnable_positional_embeddings = self.learnable_positional_embeddings(positions)
    
    # Transformer layers
    hidden = encoded_inputs + fixed_positional_embeddings + learnable_positional_embeddings
    for transformer_layer in self.transformer_layers:
      hidden = transformer_layer(hidden, global_attention, local_attention)
      
    # Output layer
    logits = self.output_layer(hidden)
    
    return logits

```

Define the TransformerLayer class, which represents a single transformer layer in the Informer model. Here is an example of how this class could be implemented:
> To define the TransformerLayer class in the Informer model, you will need to implement the forward pass of the layer. The forward pass of a transformer layer typically includes a multi-head attention function and a feed-forward neural network, followed by layer normalization and residual connections.

```python
class TransformerLayer(nn.Module):
  def __init__(self, hidden_size, num_heads):
    super(TransformerLayer, self).__init__()
    
    # Multi-head attention
    self.attention = MultiHeadAttention(hidden_size, num_heads)
    
    # Feed-forward neural network
    self.feed_forward = nn.Linear(hidden_size, hidden_size)
    
    # Layer normalization
    self.layer_norm_attention = nn.LayerNorm(hidden_size)
    self.layer_norm_feed_forward = nn.LayerNorm(hidden_size)
    
  def forward(self, inputs, global_attention, local_attention):
    # Multi-head attention
    attention_output = self.attention(inputs, inputs, inputs, global_attention, local_attention)
    attention_output = self.layer_norm_attention(inputs + attention_output)
    
    # Feed-forward neural network
    feed_forward_output = self.feed_forward(attention_output)
    feed_forward_output = self.layer_norm_feed_forward(attention_output + feed_forward_output)
    
    return feed_forward_output

```

Define the MultiHeadAttention class:

```python
class MultiHeadAttention(nn.Module):
  def __init__(self, hidden_size, num_heads):
    super(MultiHeadAttention, self).__init__()
    
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    
    self.query_transform = nn.Linear(hidden_size, hidden_size)
    self.key_transform = nn.Linear(hidden_size, hidden_size)
    self.value_transform = nn.Linear(hidden_size, hidden_size)
    self.output_transform = nn.Linear(hidden_size, hidden_size)
    
  def forward(self, query, key, value, mask=None):
    # Split the input into multiple heads
    query = self.query_transform(query).view(batch_size, -1, self.num_heads, self.hidden_size // self.num_heads)
    key = self.key_transform(key).view(batch_size, -1, self.num_heads, self.hidden_size // self.num_heads)
    value = self.value_transform(value).view(batch_size, -1, self.num_heads, self.hidden_size // self.num_heads)
    
    # Transpose and reshape the inputs
    query = query.transpose(1, 2).contiguous().view(batch_size * self.num_heads, -1, self.hidden_size // self.num_heads)
    key = key.transpose(1, 2).contiguous().view(batch_size * self.num_heads, -1, self.hidden_size // self.num_heads)
    value = value.transpose(1, 2).contiguous().view(batch_size * self.num_heads, -1, self.hidden_size // self.num_heads)
    
    # Compute the attention scores
    attention_scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(self.hidden_size // self.num_heads)
    if mask is not None:
      attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
    attention_weights = F.softmax(attention_scores, dim=-1)
    
    # Compute the weighted sum of the values
    attention_output = torch.bmm(attention_weights, value)
    
    # Reshape and transpose the output
    attention_output = attention_output.view(batch_size, self.num_heads, -1, self.hidden_size // self.num_heads).transpose(1, 2).contiguous()
    
    # Combine the attention heads
    attention_output = attention_output.view(batch_size, -1, self.hidden_size)
    
    # Apply the output transform
    attention_output = self.output_transform(attention_output)
    
    return attention_output

```

Train. 

```python
# Define the model and optimizer
model = Informer(input_size, hidden_size, num_layers, num_heads, num_classes)
optimizer = torch.optim.Adam(model.parameters())

# Define the training loop
for epoch in range(num_epochs):
  for inputs, labels in train_dataloader:
    # Forward pass
    logits = model(inputs)
    loss = F.cross_entropy(logits, labels)
    
    # Coverage loss
    attention_weights = model.transformer_layers[-1].attention.attention_weights
    coverage_loss = torch.sum(torch.min(attention_weights, model.previous_attention_weights))
    loss += coverage_weight * coverage_loss
    model.previous_attention_weights = attention_weights
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

```