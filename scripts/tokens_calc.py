import pandas as pd
import tiktoken
from datasets import load_dataset

def main():
    model = 'gpt-4' # 'gpt-3.5' encoder is same as 'gpt-4'

    encoder = tiktoken.encoding_for_model(model)

    ajgt_tokens = get_ajgt_tokens(encoder)
    apb_tokens = get_apb_tokens(encoder)
    easc_tokens = get_easc_tokens(encoder)
    padt_tokens = get_padt_tokens(encoder)
    unv1_tokens = get_unv1_tokens(encoder)
    bolt_tokens = get_bolt_tokens(encoder)

    print(f'EASC: {easc_tokens:,}')
    print(f'AJGT: {ajgt_tokens:,}')
    print(f'PADT: {padt_tokens:,}')
    print(f'APB: {apb_tokens:,}')
    print(f'UNv1: {unv1_tokens:,}')
    print(f'BOLT: {bolt_tokens:,}')
    print(f'Total: {ajgt_tokens + apb_tokens + easc_tokens + padt_tokens  + unv1_tokens + bolt_tokens}')


def get_bolt_tokens(encoder):
    base_path = "splits_ldc"
    split = 'test'
    li = []
    for filename in [f'{split}-source.arabizi', f'{split}-sentence-aligned-target.gold']:
        df = pd.read_csv(f"{base_path}/{split}/{filename}", header=None, delimiter = "\\n", engine = "python")
        li.append(df)

    df_test = pd.concat(li, axis=1, ignore_index=True)
    df_test.columns = ["arabizi", "arabic"]
    tokens_count = 0

    for element in df_test['arabizi']:
        tokens_count += len(encoder.encode(element.strip()))

    return tokens_count


def get_ajgt_tokens(encoder):
    df = pd.read_csv(
        'https://raw.githubusercontent.com/UBC-NLP/marbert/main/examples/UBC_AJGT_final_shuffled_test.tsv',
        delimiter='\t'
    )

    tokens_count = 0

    for element in df['content']:
        tokens_count += len(encoder.encode(element.strip()))

    return tokens_count


def get_apb_tokens(encoder):
    df = pd.read_csv(
        'https://raw.githubusercontent.com/marwah2001/Arabic-Paraphrasing-Benchmark/main/Arabic%20paraphrasing%20benchmark-Marwah-Alian.csv',
        delimiter=';'
    )

    tokens_count = 0

    for element in df['First sentence']:
        tokens_count += len(encoder.encode(element.strip()))

    return tokens_count


def get_easc_tokens(encoder):
    data = load_dataset('arbml/EASC')

    tokens_count = 0

    for element in data['train']['article']:
        tokens_count += len(encoder.encode(element.strip()))

    return tokens_count


def get_metrec_tokens(encoder):
    data = load_dataset('metrec')

    tokens_count = 0

    for element in data['test']['text']:
        tokens_count += len(encoder.encode(element.strip()))

    return tokens_count


def get_padt_tokens(encoder):
    data = load_dataset('universal_dependencies', 'ar_padt')

    tokens_count = 0

    for element in data['test']['text']:
        tokens_count += len(encoder.encode(element.strip()))

    return tokens_count

def get_unv1_tokens(encoder):
    with open('testsets/testset/UNv1.0.testset.ar') as fp:
        lines = list(map(str.strip, fp))

    tokens_count = 0

    for element in lines:
        tokens_count += len(encoder.encode(element.strip()))

    return tokens_count


if __name__ == '__main__':
    main()