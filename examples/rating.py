import taqyim as tq

def post_process(example):
    return example.replace('ـ', '').replace('-', '').strip()

def map_prompt(sample):
    text = f"Instruction: {sample['instruction']} Generated Text:"
    
    for l,model in zip(["A", "B", "C"], ["CIDAR", "Chat", "AlpaGasus"]):
        text += f" {l}. {post_process(sample[model])}"
    sample["text"] = text 
    return sample

prompt = """
You are given an Instruction and the Generated Text from three different models A B and C.
Choose one response that best represents the Arabic region. Respond ONLY with the letters A B or C. Don't generate any other text.
""".strip()

pipeline = tq.Pipeline(
    eval_name = "cidar-test",
    dataset_name="arbml/cidar_alpag_chat",
    task_class= "rater",
    task_description = "Arabic text sentiment analysis",
    input_column_name = 'text',
    prompt=prompt,
    api_key='<openai-key>',
    preprocessing_fn=map_prompt,
    train_split="train",
    test_split="train",
    model_name = "gpt-3.5-turbo-0301",
    max_samples= 100,)

pipeline.run()
print(pipeline.show_results())