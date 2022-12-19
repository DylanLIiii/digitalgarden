---
{"dg-publish":true,"permalink":"/personal-page/positional-encoding/"}
---

Positional encoding is a technique used in deep learning models to incorporate the relative position of input tokens into the model's representation of the input data. This is particularly important in natural language processing tasks, where the order of the input tokens can carry important meaning.

It is often used in conjunction with self-attention mechanisms, which allow the model to weight the importance of different input features based on their relationships with each other. This helps the model make more accurate predictions and allows it to model long-range dependencies in sequential data.

At a high level, positional encoding works by adding a set of learnable parameters to the input representation for each token, which encode the relative position of the token in the input sequence. These learnable parameters are typically initialized randomly and are updated during training along with the other model parameters.

One way to implement positional encoding is to use sinusoidal functions, as follows:

$$\text{PE}(pos, 2i) = \sin(\frac{pos}{10000^{2i/d_\text{model}}})$$
$$ \text{PE}(pos, 2i+1) = \cos(\frac{pos}{10000^{2i/d_\text{model}}}) $$

where $pos$ is the position of the token in the input sequence, $i$ is the dimension of the positional encoding, and $d_\text{model}$ is the number of dimensions in the model.

To implement this in python, you can use the following code:

```python
import numpy as np

def positional_encoding(sequence_length, d_model):
  # create a list of positions
  positions = np.arange(sequence_length)[:, np.newaxis]
  
  # compute the positional encoding for each position
  encoding = np.sin(positions / (10000**(2 * np.arange(d_model)[np.newaxis, :] / d_model)))
  encoding = np.concatenate([np.cos(positions / (10000**(2 * np.arange(d_model)[np.newaxis, :] / d_model))), encoding], axis=-1)
  
  return encoding

```

The output of the positional_encoding function will be a matrix with shape (sequence_length, d_model) containing the positional encoding for each position in the sequence.

### Using Positional Encoding with Self-Attention

Positional encoding is often used in conjunction with self-attention mechanisms, which allow the model to weight the importance of different input features based on their relationships with each other. To incorporate positional encoding into a self-attention mechanism, you can simply concatenate the positional encoding with the input representation for each token before passing it through the self-attention layer.

For example, suppose you have an input sequence of tokens with embeddings of shape (batch_size, sequence_length, d_model). You can incorporate positional encoding into the input representation as follows:

```python 
import numpy as np

def incorporate_positional_encoding(input_embeddings, sequence_length, d_model):
  # compute the positional encoding
  positional_encoding = positional_encoding(sequence_length, d_model)
  
  # add the positional encoding to the input embeddings
  input_embeddings += positional_encoding[np.newaxis, :, :]
  
  return input_embeddings

input_embeddings = ... # input embeddings with shape (batch_size, sequence_length, d_model)
sequence_length = ... 

```


## Another Approach 
Another approach to positional encoding is to use learned embeddings, which are trained along with the rest of the model. This can be represented mathematically as follows:

$$ PE_{pos} = W_{PE} \cdot pos $$

Where $PE_{pos}$ is the positional encoding for the word at position $pos$, and $W_{PE}$ is a learnable weight matrix.

Self-attention mechanisms allow a model to attend to different parts of the input sequence at different times, enabling it to make use of long-range dependencies and contextual information. Mathematically, self-attention can be represented as follows:

$$ Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V $$

Where $Q$, $K$, and $V$ are the query, key, and value matrices, respectively, and $d_k$ is the dimensionality of the keys.

Multi-head attention is a variant of self-attention that uses multiple attention heads in parallel, allowing the model to attend to different parts of the input sequence simultaneously. Mathematically, multi-head attention can be represented as follows:

$$ MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O $$ $$ where \ head_i = Attention(QW_i^Q, KW_i^K, VW_i^V) $$

Where $h$ is the number of attention heads, and $W_i^Q$, $W_i^K$, $W_i^V$, and $W^O$ are learnable weight matrices.
```python
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, activation):
        super(Transformer, self).__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation), num_encoder_layers, n

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")

        # Calculate the positional encoding
        src_pos = torch.arange(0, src.size(1)).unsqueeze(1).expand(src.size(1), src.size(0)).to(src.device)
        tgt_pos = torch.arange(0, tgt.size(1)).unsqueeze(1).expand(tgt.size(1), tgt.size(0)).to(tgt.device)

        src_pos = self.positional_encoding(src_pos)
        tgt_pos = self.positional_encoding(tgt_pos)

        # Add the positional encoding to the input embeddings
        src = src + src_pos
        tgt = tgt + tgt_pos

        # Pass the input through the encoder and decoder
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        memory = self.encoder(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        return output

# Define the loss function
loss_fn = nn.CrossEntropyLoss()

# Choose an optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(batch.src, batch.tgt)

        # Calculate the loss
        loss = loss_fn(output, batch.label)

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()



```