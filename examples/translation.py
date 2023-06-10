import taqyim as tq

def map_labels(sample):
    if sample["label"] == "1":
        sample["label"] = "Positive"
    else:
        sample["label"] = "Negative"
    return sample

prompt = """
Translate the following statement from Arabic to Russian
""".strip()

pipeline = tq.Pipeline(
            eval_name = "tatoeba-test",
            dataset_name= "Helsinki-NLP/tatoeba_mt",
            subset = "ara-rus",
            task_class= "translation",
            task_desc = "Translation from Russian to Arabic",
            input_column_name = 'sourceString',
            target_column_name = 'targetString',
            prompt=prompt,
            api_key='<openai-key>',
            train_split="validation",
            test_split="test",
            model_name = "gpt-3.5-turbo-0301",
            resume_from_record= True,
            max_samples= 2,)
pipeline.run()
print(pipeline.show_results())