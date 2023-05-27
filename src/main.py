import pandas as pd
import datasets
import json

"""
results = pipeline(
    task = str,
    dataset_path = str,
    train_split = str,
    test_split = str,
    columns_name = list,
    preprocessing = fn,
    fewshots = int:0,
    prompt = str, # jinja
    max_samples = int:-1, # by default, the length of the test split
    threads = int:1, # by default 1
    threads_timeout = int:100, # by default 100
    records_path = str, # by default ./records
    resume_from_records = bool, # True
    temperature = int:0, # by default 0
)
"""


def pipeline(
    dataset_name="arbml/easc",
    preprocessing_fn=None,
    train_split="train",
    test_split=None,
    records_path="easc.jsonl",
):
    train_dataset = datasets.load_dataset(dataset_name, split=train_split)
    if test_split is not None:
        test_dataset = datasets.load_dataset(dataset_name, split=test_split)

    # preprocess the dataset
    if preprocessing_fn is not None:
        train_dataset = train_dataset.map(preprocessing_fn)
        test_dataset = test_dataset.map(preprocessing_fn)

    resume_from_records = True
    if resume_from_records:
        success_ids = get_success_record_ids(records_path=records_path)
        unsuccess_ids = set(range(len(test_dataset))) - set(success_ids)
        test_dataset = test_dataset.select(unsuccess_ids)
    print(test_dataset)


def get_success_record_ids(records_path):
    with open(records_path, "r") as records_file:
        records_str = records_file.read().splitlines()
        records = [json.loads(record_str) for record_str in records_str]
    records = list(filter(lambda record: "type" in record.keys(), records))
    success_type = "summarization"
    success_records = list(
        filter(
            lambda record: record["type"] == success_type,
            records,
        )
    )
    success_records = sorted(
        success_records,
        key=lambda record: int(record["sample_id"].split(".")[-1]),
    )
    success_ids = [
        int(record["sample_id"].split(".")[-1]) for record in success_records
    ]
    return sorted(set(success_ids))
