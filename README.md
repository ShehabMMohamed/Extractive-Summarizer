# Extractive Summarization with BERT

## Usage
[![](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)]()
```python
import torch
from models.model_builder import ExtSummarizer
from ext_sum import summarize

# Load model -> ['bertbase', 'distilbert', 'squeezebert', 'mobilebert']
model_type = 'squeezebert'
checkpoint = torch.load(f'checkpoints/{model_type}_ext.pt', map_location='cpu')
model = ExtSummarizer(checkpoint=checkpoint, bert_type=model_type, device='cpu')

# Run summarization
input_fp = 'raw_data/input.txt'
result_fp = 'results/summary.txt'
summary = summarize(input_fp, result_fp, model, max_length=3)
print(summary)
```

