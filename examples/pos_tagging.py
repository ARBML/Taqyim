def map_fn(sample):
    TAGS = ["NOUN","PUNCT","ADP","NUM","SYM","SCONJ","ADJ","PART","DET","CCONJ","PROPN","PRON","X","_", "ADV","INTJ","VERB","AUX"]
    tag_names_to_labels = {tag_name: tag_label for tag_label, tag_name in enumerate(TAGS)}
    tag_labels_to_names = {
        str(tag_label): tag_name for tag_name, tag_label in tag_names_to_labels.items()
    }
    mapped_tags = []
    for id_ in sample["upos"]:
        mapped_tags.append(tag_labels_to_names[id_])
    
    sample["upos"] = mapped_tags
    tags_map = ""
    tokens = []
    for token, tag in zip(sample['tokens'], sample["upos"]):
        if tag == '_': 
            continue
        tags_map += f"{token}:{tag}"
        tags_map += "\n"
        tokens.append(token)

    sample["upos"] = tags_map
    sample['tokens'] = ' '.join(tokens)
    return sample

prompt = """
I wish you can generate a table of Arabic POS tags following Universal Dependencies tagset in the following format:
TOKEN:POS
Please note that I tokenized the sentence for you. Please do not change, add, combine, merge or remove any of these tokens such as ب and ه. Please consider punctuation marks as separate tokens, always. Split them as two separate tokens if they come together and classify each of them independently.
Please give me the generated table and that is it. No further discussion, explanation or extrapolation required.
""".strip()

pipeline(
    eval_name = "padt-test",
    task_class= "pos_tagging",
    task_description = "Arabic text PoS tagging",
    input_column_name = 'tokens',
    target_column_name = 'upos',
    prompt=prompt,
    api_key='<openai-key>',
    dataset_name="universal_dependencies",
    preprocessing_fn=map_fn,
    train_split="train",
    test_split="test",
    threads = 1,
    threads_timeout=100,
    model_name = "gpt-4-0314",
    temperature = 0.0,
    resume_from_record = False,
    max_samples= 680,
    subset= "ar_padt",
)