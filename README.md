# Tqeem تقييم

An example for resuming an eval ...

```python

def map_labels(sample):
    if sample["label"] == "1":
        sample["label"] = "Positive"
    else:
        sample["label"] = "Negative"
    return sample

pipeline(
    eval_name = "ajgt2",
    task_class= "classification",
    task_description = "Arabic text classification",
    input_column_name = 'text',
    target_column_name = 'label',
    prompt='What is the sentiment?',
    api_key='<api-key>',
    dataset_name="ajgt_twitter_ar",
    preprocessing_input_fn=None,
    preprocessing_target_fn=map_labels,
    train_split="train",
    test_split="train",
    threads = 1,
    threads_timeout=100,
    model_name = "gpt-3.5-turbo-0301",
    temperature = 0.0,
    resume_from_record = True,
    max_samples= 3
)
```