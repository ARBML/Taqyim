import pandas as pd
import datasets
import json
import os
from pathlib import Path
from datasets import Value, Sequence
import pandas as pd

metrics = {'classification':['accuracy'], 'tagging':['accuracy'], 'translate':['sacrebleu_sentence_score']}
task_eval_names = {'pos_tagging': 'tagging', 'classification':'classification', 'translate':'metrics'}
BASE_PATH = Path.home()/".evals"


class Pipeline:
    def __init__(
        self,
        eval_name, 
        task_class,   
        input_column_name,
        target_column_name,
        prompt,
        api_key,
        dataset_name,
        preprocessing_fn=None,
        train_split="train",
        test_split=None,
        threads = 1,
        threads_timeout=100,
        max_samples = -1,
        model_name = "gpt-3.5-turbo-0301",
        temperature = 0.0,
        task_description ='',
        num_few_shot=0,
        resume_from_record = False,
        subset = None,
        seed = 41,
    ):
        self.eval_name = eval_name 
        self.task_class = task_class   
        self.input_column_name = input_column_name
        self.target_column_name = target_column_name
        self.prompt = prompt
        self.api_key = api_key
        self.dataset_name = dataset_name
        self.preprocessing_fn = preprocessing_fn
        self.train_split = train_split
        self.test_split= test_split
        self.threads = threads
        self.threads_timeout= threads_timeout
        self.max_samples = max_samples
        self.model_name = model_name
        self.temperature = temperature
        self.task_description = task_description
        self.num_few_shot = num_few_shot
        self.resume_from_record = resume_from_record
        self.subset = subset
        self.seed = seed

        os.system(f'mkdir -p {BASE_PATH}')
        os.system(f'mkdir -p {BASE_PATH}/evals')
        os.system(f'mkdir -p {BASE_PATH}/eval_results/')
        self.record_path = f"{BASE_PATH}/eval_results/{self.eval_name}.jsonl"
        self.specs_file = f'{BASE_PATH}/evals/{self.eval_name}.yaml'


        os.system(f'mkdir -p {BASE_PATH}/data/{self.eval_name}')
        data_path = f'{BASE_PATH}/data/{self.eval_name}'
        
        if self.subset is not None:
            train_dataset = datasets.load_dataset(self.dataset_name, self.subset, split=self.train_split)
            if self.test_split is not None:
                test_dataset = datasets.load_dataset(self.dataset_name, self.subset, split=self.test_split)
        else:
            train_dataset = datasets.load_dataset(self.dataset_name, split=self.train_split)
            if test_split is not None:
                test_dataset = datasets.load_dataset(self.dataset_name, split=self.test_split)
        
        if self.max_samples == -1:
            self.max_samples = len(test_dataset)
        
        test_dataset = test_dataset.select(range(self.max_samples))
        self.unsuccess_ids = []

        if resume_from_record:
            print('Trying to resume the run ... ')
            success_ids = self.get_success_record_ids(records_path=self.record_path, task_type=task_eval_names[self.task_class.lower()])
            self.unsuccess_ids = sorted(set(range(len(test_dataset))) - set(success_ids))

            test_dataset = test_dataset.select(self.unsuccess_ids)
            if len(test_dataset) == 0:
                raise('Run already finished ...')

            self.record_path = f"{self.record_path}_resume"

        # Convert the input and target features
        train_dataset = self.cast_features(train_dataset, features = [self.input_column_name, self.target_column_name])
        test_dataset = self.cast_features(test_dataset, features = [self.input_column_name, self.target_column_name])    
        
        if self.preprocessing_fn is not None:
            train_dataset = train_dataset.map(self.preprocessing_fn)
            test_dataset = test_dataset.map(self.preprocessing_fn)

        dev_df = train_dataset.to_pandas()
        dev_df["sample"] = dev_df.apply(lambda x: self.create_chat_example(x), axis=1)
        dev_df[["sample"]].to_json(f'{data_path}/few_shot.jsonl', lines=True, orient="records",force_ascii=False)

        test_df = test_dataset.to_pandas()
        test_df["input"] = test_df[self.input_column_name].apply(lambda x: self.create_chat_prompt(x))
        test_df["ideal"] = test_df[self.target_column_name]
        test_df[["input", "ideal"]].to_json(f'{data_path}/samples.jsonl', lines=True, orient="records",force_ascii=False)

        os.environ["OPENAI_API_KEY"] = self.api_key
        os.environ["EVALS_THREADS"]=f"{self.threads}"
        os.environ["EVALS_THREAD_TIMEOUT"]=f"{self.threads_timeout}"
        
        specs = f"""
    {self.eval_name}:
        id: {self.eval_name}.test.v1
        description: {self.task_description}
{self.eval_name}.test.v1:
    class: evals.elsuite.{self.task_class.lower()}:{self.task_class.lower().title()}
    args:
        samples_jsonl: {BASE_PATH}/data/{self.eval_name}/samples.jsonl
        num_few_shot: {self.num_few_shot} # max few shots to use
        """.strip()


        with open(self.specs_file, "w") as file:
            file.write(specs)

    def run(self):        
        os.system(f"oaieval {self.model_name} {self.eval_name}\
                    --seed {self.seed} \
                    --modelspec_extra_options temperature={self.temperature} --max_samples {self.max_samples} --record_path {self.record_path}")

        if self.resume_from_record:
            self.merged_record_path = f'{BASE_PATH}/eval_results/{self.eval_name}_full.jsonl'
            print('Merging evals to', self.merged_record_path)
            eval_paths = [self.record_path.split('resume')[0][:-1], self.record_path]
            records = self.merge_evals(task_type=task_eval_names[self.task_class], evals_record_paths=eval_paths)
            with open(self.merged_record_path, 'w') as f:
                for sample in records:
                    f.write(json.dumps(sample, ensure_ascii = False) + "\n")
    
    def show_results(self):
        if self.resume_from_record:
            with open(self.merged_record_path, "r") as f:
                events_df = pd.read_json(f, lines=True)
        else:
            with open(self.record_path, "r") as f:
                events_df = pd.read_json(f, lines=True)
        
        prompts = []
        samples = []
        
        for i, r in pd.json_normalize(events_df[events_df.type == "sampling"].data).iterrows():
            prompts.append(r.prompt[-1]['content'])
            samples.append(r.sampled)
        return pd.DataFrame({'prompt': prompts, 'Sample':samples})

    def create_chat_prompt(self, x):
        return [
            {"role": "system", "content": self.prompt}, 
            {"role": "user", "content": x}
        ]

    def create_chat_example(self, x):
        return [
            {"role": "system", "content": x[self.input_column_name], "name": "example_user"},
            {"role": "system", "content": x[self.target_column_name], "name": "example_assistant"},
        ]

    def merge_evals(self, evals_record_paths =[], task_type = ""):
        out_records = []
        curr_sample_id = 0
        curr_event_id = 0
        for record_id, self.record_path in enumerate(evals_record_paths):
            with open(self.record_path, "r") as records_file:
                records_str = records_file.read().splitlines()
                records = [json.loads(record_str) for record_str in records_str]

                #save the spec records for the current json
                spec_records = list(filter(lambda record: "spec" in record.keys(), records)) #extract records with types
                for record in spec_records:
                    out_records.append(record)

                records = list(filter(lambda record: "type" in record.keys(), records)) #extract records with types
                task_records = list(filter(lambda record: record["type"] == task_type, records)) #extract records with types
                
                
                for i in range(len(task_records)):
                    # filter the records containing the sample id
                    sample_id_records = list(filter(lambda record: record["sample_id"] == f'{self.eval_name}.test.{i}', records))
                    # print(sample_id_records)
                    sample_id_record_copy = {}
                    for sample_id_record in sample_id_records:
                        sample_id_record_copy = dict(sample_id_record) # don't modify a dict inside a for loop 
                        if record_id > 0:
                            if i < len(self.unsuccess_ids):
                                sample_id_record_copy['sample_id'] = f"{self.eval_name}.test.{self.unsuccess_ids[i]}"
                            else:
                                sample_id_record_copy['sample_id'] = f"{self.eval_name}.test.{curr_sample_id}"
                        else:
                            sample_id_record_copy['sample_id'] = f"{self.eval_name}.test.{i}"

                        sample_id_record_copy['event_id'] = curr_event_id
                        curr_event_id += 1
                        out_records.append(sample_id_record_copy)

                    if len(sample_id_records) > 1:
                        curr_sample_id += 1
                            

        final_report =  {metric: 0.0 for metric in metrics[self.task_class]}
        task_metrics = metrics[self.task_class]
        for record in out_records:
            if 'type' in record and record['type'] == task_type:
                for metric in task_metrics:
                    final_report[metric] += record["data"][metric] / curr_sample_id # what happens if there are repeated samples ?:

        out_records.append({'final_report':final_report})
        return out_records

    def get_success_record_ids(self, records_path, task_type = ""):

        with open(records_path, "r") as records_file:
            records_str = records_file.read().splitlines()
            records = [json.loads(record_str) for record_str in records_str]
        records = list(filter(lambda record: "type" in record.keys(), records))
        success_records = list(
            filter(
                lambda record: record["type"] == task_type,
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

    def cast_features(self, dataset, features= []):
        new_features = dataset.features.copy()
        lst_features = []
        for feature in features:
            if isinstance(dataset[feature][0], list):
                new_features[feature] = Sequence(Value(dtype='string', id=None))
            else:
                new_features[feature] = Value(dtype='string', id=None)
        dataset = dataset.cast(new_features)
        return dataset

