# Taqyim تقييم

A library for evaluting Arabic NLP datasets on chatgpt models. 

## Installation

```
pip install -e .
```

## Example 

```python
import taqyim as tq

# map the labels to positive and negative
def map_labels(sample):
    if sample["label"] == "1":
        sample["label"] = "Positive"
    else:
        sample["label"] = "Negative"
    return sample

# create the eval class
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

# run the evaluation
pipeline.run()

# show the results
pipeline.show_results()

# show the eval metrics
pipeline.get_final_report()

```

## Run on custom dataset

[custom_dataset.ipynb](custom_dataset.ipynb) has a complete example on how to run evaluation on a custom dataset. 


## parameters

    *    `eval_name` choose an eval name
    *    `task_class` class name from supported class names
    *    `dataset_name` dataset name for evaluation
    *    `subset` If the dataset has subset
    *    `train_split` train split name in the dataset
    *    `test_split`test split name in the dataset
    *    `input_column_name` input column name in the dataset
    *    `target_column_name` target column name in the dataset
    *    `prompt` the prompt to be fed to the model
    *    `task_description` short string explaining the task
    *    `api_key` api key from [keys](https://platform.openai.com/account/api-keys)
    *    `preprocessing_fn` function used to process inputs and targets 
    *    `threads` number of threads used to fetch the api
    *    `threads_timeout` thread timeout 
    *    `max_samples` max samples used for evaluation from the dataset 
    *    `model_name` choose either `gpt-3.5-turbo-0301` or `gpt-4-0314`
    *    `temperature` temperature passed to the model between 0 and 2, higher temperature means more random results
    *    `num_few_shot` number of fewshot samples to be used for evaluation
    *    `resume_from_record` if `True` it will continue the run from the sample that has no results. 
    *    `seed` seed to redproduce the results

## Supported Classes and Tasks

* `Classification` classification tasks see [classification.py](examples/classification.py).
* `Pos_Tagging` part of speech tagging tasks [pos_tagging.py](examples/pos_tagging.py).
* `Translation` machine translation [translation.py](examples/translation.py).
* `Summarization` machine translation [summarization.py](examples/summarization.py).
* `Diacritization` machine translation [diacritization.py](examples/diacritization.py).
