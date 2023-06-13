import taqyim as tq

prompt = 'Respond only positive or negative sentiment. Respond only with the class name in English:'
temps = [0, 0.5, 1.0, 1.5, 2.0]

for temp in temps:

    pipeline = tq.Pipeline(
        eval_name = f"ajgt-temp-gpt-4-{temp}",
        dataset_name="arbml/ajgt_ubc_split",
        task_class= "classification",
        task_desc = "Arabic text sentiment analysis",
        input_column_name = 'content',
        target_column_name = 'label',
        prompt=prompt,
        api_key='<openai-key>',
        train_split="train",
        test_split="test",
        model_name = "gpt-4-0314",
        temperature= temp,
        max_samples= 360,)

    pipeline.run()
    print(pipeline.show_results())