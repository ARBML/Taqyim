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

pipeline = tq.Pipeline(
    eval_name = "ajgt-test",
    dataset_name="ajgt_twitter_ar",
    task_class= "classification",
    task_description = "Arabic text sentiment analysis",
    input_column_name = 'text',
    target_column_name = 'label',
    prompt="Predict the sentiment of the following statement in Arabic",
    api_key='<openai-key>,
    preprocessing_fn=map_labels,
    train_split="train",
    test_split="train",
    model_name = "gpt-3.5-turbo-0301",
    max_samples= 1,)

pipeline.run()

print(pipeline.show_results())
```

## Supported Classes and Tasks

* `Classification` classification tasks see [classification.py](examples/classification.py).
* `Pos_Tagging` part of speech tagging tasks [pos_tagging.py](examples/pos_tagging.py).
* `Translate` machine translation [translation.py](examples/translation.py).