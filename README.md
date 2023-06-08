# Taqyim تقييم

A library for evaluting Arabic NLP datasets on chatgpt models. 

## Installation

```
pip install -e .
```

## Example 

```python
import taqyim as tq

def map_labels(sample):
    if sample["label"] == "1":
        sample["label"] = "Positive"
    else:
        sample["label"] = "Negative"
    return sample

prompt = """
Predict the sentiment of the following statement in Arabic
""".strip()

tq.pipeline(
    eval_name = "ajgt-test",
    dataset_name="ajgt_twitter_ar",
    task_class= "classification",
    task_description = "Arabic text sentiment analysis",
    input_column_name = 'text',
    target_column_name = 'label',
    prompt=prompt,
    api_key='<openai-key>,
    preprocessing_fn=map_labels,
    train_split="train",
    test_split="train",
    threads = 1,
    threads_timeout=100,
    model_name = "gpt-3.5-turbo-0301",
    temperature = 0.0,
    resume_from_record = False,
    max_samples= 1,)
```

## Supported Classes and Tasks
* `Classification` classification tasks see [examples](examples/classification.py)
* `Pos_Tagging` part of speech tagging tasks [examples](examples/pos_tagging.py)
* `Translate` machine translation [examples](examples/translation.py)