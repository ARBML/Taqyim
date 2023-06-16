# Taqyim تقييم

<p align="center">
    <img width = "150px" src="https://github.com/ARBML/Taqyim/assets/15667714/6710535a-4d0b-4c1a-8c35-49b2e2110600"></img>
</p>

A library for evaluting Arabic NLP datasets on chatgpt models. 

## Installation

```
pip install -e .
```

## Example 

```python
import taqyim as tq
pipeline = tq.Pipeline(
    eval_name="ajgt-test",
    dataset_name="arbml/ajgt_ubc_split",
    task_class="classification",
    task_description= "Sentiment Analysis",
    input_column_name="content",
    target_column_name="label",
    prompt="Predict the sentiment",
    api_key="<openai-key>",
    train_split="train",
    test_split="test",
    model_name="gpt-3.5-turbo-0301",
    max_samples=1,)

# run the evaluation
pipeline.run()

# show the output data frame
pipeline.show_results()

# show the eval metrics
pipeline.get_final_report()

```

## Run on custom dataset

[custom_dataset.ipynb](notebooks/custom_dataset.ipynb) has a complete example on how to run evaluation on a custom dataset. 


## parameters

-    `eval_name` choose an eval name
-    `task_class` class name from supported class names
-    `task_desc` short description about the task
-    `dataset_name` dataset name for evaluation
-    `subset` If the dataset has subset
-    `train_split` train split name in the dataset
-    `test_split`test split name in the dataset
-    `input_column_name` input column name in the dataset
-    `target_column_name` target column name in the dataset
-    `prompt` the prompt to be fed to the model
-    `task_description` short string explaining the task
-    `api_key` api key from [keys](https://platform.openai.com/account/api-keys)
-    `preprocessing_fn` function used to process inputs and targets 
-    `threads` number of threads used to fetch the api
-    `threads_timeout` thread timeout 
-    `max_samples` max samples used for evaluation from the dataset 
-    `model_name` choose either `gpt-3.5-turbo-0301` or `gpt-4-0314`
-    `temperature` temperature passed to the model between 0 and 2, higher temperature means more random results
-    `num_few_shot` number of fewshot samples to be used for evaluation
-    `resume_from_record` if `True` it will continue the run from the sample that has no results. 
-    `seed` seed to redproduce the results

## Supported Classes and Tasks

* `Classification` classification tasks see [classification.py](examples/classification.py).
* `Pos_Tagging` part of speech tagging tasks [pos_tagging.py](examples/pos_tagging.py).
* `Translation` machine translation [translation.py](examples/translation.py).
* `Summarization` machine translation [summarization.py](examples/summarization.py).
* `Diacritization` machine translation [diacritization.py](examples/diacritization.py).

# Evaluation on Arabic Tasks 

|Tasks              |Dataset        |Size       |Metrics	    |GPT-3.5 	    |GPT-4      |SoTA|
| :---              | :---:         | :---:     | :---:         | :---:         | :---:     |:---:|
|Summarization      |[EASC](https://huggingface.co/datasets/arbml/EASC)	        |153	    |RougeL	        |23.5		    |18.25	    |62.98|
|PoS Tagging	    |[PADT](https://huggingface.co/datasets/universal_dependencies/viewer/ar_padt/train)	        |680        |Accuracy	    |75.91		    |86.29	    |97.00|
|classification	    |[AJGT](https://huggingface.co/datasets/ajgt_twitter_ar)	        |360        |Accuracy	    |87.77		    |91.13	    |96.11|	
|transliteration	|[BOLT Egyptian](https://catalog.ldc.upenn.edu/LDC2021T17)✢  |6,653      |BLEU           |13.76		    |27.66	    |58.70|
|translation	    |[UN v1](https://drive.google.com/file/d/13GI1F1hvwpMUGBSa0QC6ov4eE57GC_Zx/view)          |4,000	    |BLEU	        |35.05		    |38.83	    |53.29|
|Paraphrasing	    |[APB](https://github.com/marwah2001/Arabic-Paraphrasing-Benchmark)	        |1,010      |BLEU	        |4.295		    |6.104	    |17.52|
|Diacritization	    |[WikiNews]()✢✢      |18,286	    |WER/DER	    |32.74/10.29	| 38.06/11.64		   |12.76/3.54|

✢ BOLT requires LDC subscription

✢✢ WikiNews not public, contact [authors](https://aclanthology.org/W17-1302/) to access the dataset
