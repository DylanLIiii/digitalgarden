---
{"dg-publish":true,"permalink":"/personal-page/transformer/"}
---

### Architecture

At a high level, Transformer consists of an encoder and a decoder, each of which is made up of multiple "layers." Each layer consists of a self-attention mechanism and a feedforward network. The self-attention mechanism allows the model to weight the importance of different input features based on their relationships with each other, which helps the model make more accurate predictions. The feedforward network then processes these weighted features and produces a set of intermediate representations.

The encoder takes in a sequence of input tokens (such as words in a sentence) and produces a set of contextualized representations for each token. The decoder takes in these contextualized representations and a sequence of output tokens (such as a translation of the input sentence) and produces a set of predicted output tokens.

The self-attention mechanism used in Transformer is known as "multi-head attention," as it involves computing multiple attention weights simultaneously and concatenating them before processing them further. This allows the model to capture different types of dependencies between the input features and make more accurate predictions.

The multi-head attention mechanism is defined as follows:

$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O $$

where $Q$, $K$, and $V$ are the query, key, and value matrices, respectively, $h$ is the number of heads, and $W^O$ is a linear transformation matrix. The attention weight for each head is computed as follows:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

where $d_k$ is the dimensionality of the keys.

```python
import numpy as np

def multi_head_attention(Q, K, V, h, d_k):
  # compute the attention weights for each head
  attention_weights = np.dot(Q, K.T) / np.sqrt(d_k)
  attention_weights = np.softmax(attention_weights, axis=1)
  
  # apply the attention weights to the values
  attention_applied = np.dot(attention_weights, V)
  
  # concatenate the attention-applied values for all heads
  multi_head_attention = np.concatenate(attention_applied, axis=1)
  
  return multi_head_attention

Q = ... # query matrix with shape (batch_size, sequence_length, d_q)
K = ... # key matrix with shape (batch_size, sequence_length, d_k)
V = ... # value matrix with shape (batch_size, sequence_length, d_v)
h = ... # number of heads
d_k = ... # key dimensionality

multi_head_attention = multi_head_attention(Q, K, V, h, d_k)


```
The output of the multi-head attention function will have shape (batch_size, sequence_length, d_q * h), where d_q * h is the total number of dimensions in the concatenated attention-applied values for all heads.

### Loss Function

The loss function used in Transformer is typically the cross-entropy loss, which is a common choice for classification tasks. The cross-entropy loss is defined as follows:

$$ L = -\frac{1}{N} \sum_{i=1}^{N} y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) $$

where $N$ is the number of examples in the training set, $y_i$ is the true label for example $i$, and $\hat{y}_i$ is the predicted probability that example $i$ belongs to the positive class. The cross-entropy loss is minimized when the predicted probabilities are close to the true labels.

```python
import numpy as np

def cross_entropy_loss(y, y_hat):
  # compute the loss for each example
  losses = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
  
  # average the loss over the batch
  loss = np.mean(losses)
  
  return loss

y = ... # true labels with shape (batch_size,)
y_hat = ... # predicted probabilities with shape (batch_size,)

loss = cross_entropy_loss(y, y_hat)


```


### Optimization

Transformer is typically trained using a variant of stochastic gradient descent (SGD) called Adam, which is a popular optimization algorithm for deep learning models. Adam works by maintaining an exponentially decaying average of the gradient and the squared gradient, which are used to adapt the learning rate on a per-parameter basis. This allows Adam to take larger steps in the direction of the gradient when the gradient is large and stable, and smaller steps when the gradient is small and fluctuating.

```python
import numpy as np

def adam(params, grads, m, v, t, learning_rate, beta_1, beta_2, epsilon):
  # update the moving averages of the gradient and squared gradient
  m = beta_1 * m + (1  -beta_1) * grads 
  v = beta_2 * v + (1 - beta_2) * np.square(grads)
  m_hat = m / (1 - beta_1**t) 
  # correct for bias in the moving averages
  v_hat = v / (1 - beta_2**t)
  # update the parameters
  params -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

  return params, m, v

```

You can then call this function with the appropriate input values:

```python
params = ... # model parameters 
grads = ... # gradients with respect to the parameters 
m = ... # first moment (moving average of gradient) 
v = ... # second moment (moving average of squared gradient) 
t = ... # current timestep (for bias correction) 
learning_rate = ... # learning rate 
beta_1 = ... # exponential decay rate for first moment 
beta_2 = ... # exponential decay rate for second moment 
epsilon = ... # small constant for numerical stability
params, m, v = adam(params, grads, m, v, t, learning_rate, beta_1, beta_2, epsilon)
```


### Key Design Decisions

There are several key design decisions that went into the development of Transformer:

1.  Self-attention mechanism: One of the key innovations in Transformer is the use of self-attention mechanisms, which allow the model to weight the importance of different input features based on their relationships with each other. This helps the model make more accurate predictions and allows it to model long-range dependencies in sequential data.
    
2.  Multi-head attention: Transformer uses a multi-head attention mechanism, which involves computing multiple attention weights simultaneously and concatenating them before processing them further. This allows the model to capture different types of dependencies between the input features and make more accurate predictions.
    
3.  Encoder-decoder architecture: Transformer uses an encoder-decoder architecture, with the encoder taking in a sequence of input tokens and producing contextualized representations for each token, and the decoder taking in these contextualized representations and a sequence of output tokens and producing predicted output tokens.
    
4.  Adam optimization: Transformer is typically trained using Adam, a variant of stochastic gradient descent that is particularly well-suited for deep learning models. Adam is able to adapt the learning rate on a per-parameter basis, which helps the model converge more quickly and achieve better performance.
    
5.  [[Personal Page/Positional encoding\|Positional encoding]]: To account for the order of the input tokens, Transformer uses a positional encoding mechanism, which adds a set of learnable parameters to the input representation for each token. This allows the model to capture the relative position of the tokens and make more accurate predictions.
    

### Comparison to Other Models

Transformer has been shown to outperform other state-of-the-art models on a variety of natural language processing tasks, including machine translation, language modeling, and summarization. One of the key advantages of Transformer is its ability to model long-range dependencies in sequential data, thanks to its use of self-attention mechanisms. This allows it to handle tasks that require a deep understanding of the context and structure of the input data, such as machine translation.

However, it's worth noting that Transformer is not always the best choice for every task. Like any machine learning model, it has its own strengths and weaknesses, and it may not perform as well on certain tasks as other models. It's always important to carefully consider the characteristics of the problem at hand and choose the most appropriate model for the task.