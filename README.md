# BERT implemented from scratch

In this file, I implemented BERT from scratch, focusing on the core concepts of transformer architecture, attention mechanisms, and matrix operations. BERT (Bidirectional Encoder Representations from Transformers) is a pivotal model in NLP that uses a bidirectional approach to understand the context of words in a sentence.

<br>

I will also demonstrate how to load tensors directly from the pre-trained BERT model provided by Hugging Face. Before running this file, you need to download the weights. Here is the official link to download the weights: [Hugging Face BERT](https://huggingface.co/bert-base-uncased)

<div>
    <img src="images/all-steps.png"/ width=800>
</div>

## Model Initialization and Input Processing

### Tokenization
I won’t be implementing a BPE tokenizer here, but Andrej Karpathy has a neat implementation that you might find useful.
<br>
You can check out his implementation at this link: https://github.com/karpathy/minbpe

```python
# Load Model
import torch
from transformers import BertForMaskedLM, BertTokenizer
model_path="bert-base-uncased"
tokenizer=BertTokenizer.from_pretrained(model_path)
tokenizer.encode("hello world!")
tokenizer.decode(tokenizer.encode("hello world!"))
```

### Loading and Inspecting Model Configuration and Weights

Here, the model configuration and weights are loaded from a pretrained model file. This step ensures that the necessary parameters are ready for the forward pass.

Normally, reading a model file depends on how the model classes are written and the variable names within them. 

# 1. Load the pre-trained model using transformers

```python
# Load model configuration and weights
model = BertForMaskedLM.from_pretrained(model_path, torch_dtype=torch.float32)
```
However, since we are implementing BERT from scratch, we will read the file one tensor at a time, carefully examining the embedding layers, attention heads, and feed-forward networks that make up BERT’s architecture.
