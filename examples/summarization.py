import taqyim as tq

prompt = """
Summarize the following article in Arabic
""".strip()

pipeline = tq.Pipeline(
    eval_name = "easc-test",
    dataset_name="arbml/easc",
    task_class= "summarization",
    task_description = "Arabic text summarization",
    input_column_name = 'article',
    target_column_name = 'summary',
    prompt=prompt,
    api_key='<openai-key>',
    train_split="train",
    test_split="train",
    model_name = "gpt-3.5-turbo-0301",
    max_samples= 2,)

pipeline.run()
print(pipeline.show_results())