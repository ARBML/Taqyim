import taqyim as tq

prompts = [
    'Predict the sentiment of the following statement in English: choose an option: Positive , Negative',
    'Is the sentiment of the following statement Positive or Negative?',
    'ماهي عاطفة الجملة التالية : أجب Positive أو Negative',
    'You are a helpful assistant that can predict whether a given statement in Arabic is Positive or Negative'
]
for i, prompt in enumerate(prompts):

    pipeline = tq.Pipeline(
        eval_name = f"ajgt-prompt-gpt-4-{i}",
        dataset_name="arbml/ajgt_ubc_split",
        task_class= "classification",
        task_description = "Arabic text sentiment analysis",
        input_column_name = 'content',
        target_column_name = 'label',
        prompt=prompt,
        api_key='<openai-key>',
        train_split="train",
        test_split="test",
        model_name = "gpt-4-0314",
        max_samples= 360,)

    pipeline.run()
    print(pipeline.show_results())