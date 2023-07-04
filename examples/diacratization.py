import taqyim as tq

prompt = """
Add diacritics to the following statement in Arabic
""".strip()

pipeline = tq.Pipeline(
    eval_name = "tashkeela-test",
    dataset_name="arbml/tashkeelav2",
    task_class= "diacritization",
    task_desc = "Arabic text diacritization",
    input_column_name = 'text',
    target_column_name = 'diacratized',
    prompt=prompt,
    api_key='<openai-key>',
    train_split="train",
    test_split="test",
    model_name = "gpt-3.5-turbo-0301",
    max_samples= 2,)
pipeline.run()
print(pipeline.show_results())
