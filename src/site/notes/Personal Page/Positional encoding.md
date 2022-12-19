---
{"dg-publish":true,"permalink":"/personal-page/positional-encoding/"}
---

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

