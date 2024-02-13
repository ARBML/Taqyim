import taqyim as tq

def map_labels(sample):
    sample['text'] = f"Question: {sample['Question']} Choices: A. {sample['A']} B. {sample['B']} C. {sample['C']} D. {sample['D']}"
    return sample

prompt = """
You are given a question and its choices. Please respond with the letter of the correct answer.
Do NOT add any extra text. 
""".strip()

pipeline = tq.Pipeline(
    eval_name = "cidar-mcq",
    dataset_name="arbml/cidar-mcq-100",
    task_class= "classification",
    task_description = "Arabic MCQ",
    input_column_name = 'text',
    target_column_name = 'answer',
    prompt=prompt,
    api_key='<openai-key>',
    preprocessing_fn=map_labels,
    train_split="train",
    test_split="train",
    model_name = "gpt-3.5-turbo-0301",
    max_samples= 100,)

pipeline.run()
print(pipeline.show_results())