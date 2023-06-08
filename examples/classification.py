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

pipeline = tq.Pipeline(
    eval_name = "ajgt-test",
    dataset_name="ajgt_twitter_ar",
    task_class= "classification",
    task_description = "Arabic text sentiment analysis",
    input_column_name = 'text',
    target_column_name = 'label',
    prompt=prompt,
    api_key='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',
    preprocessing_fn=map_labels,
    train_split="train",
    test_split="train",
    model_name = "gpt-3.5-turbo-0301",
    max_samples= 2,)

pipeline.run()
print(pipeline.show_results())