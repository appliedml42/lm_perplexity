import argparse
import os.path

import numpy as np

import lm_perplexity.models as models
import lm_perplexity.utils as utils
from lm_perplexity.save_lm_perplexity_data import compute_perplexity_data
import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--doc_indices_path', type=str, required=True)
    parser.add_argument('--wandb_project', type=str, required=True)
    parser.add_argument('--step', type=str, required=True)
    return parser.parse_args()


DATASET_NAMES_DICT = {
    "Github": "github",
    "ArXiv": "arxiv",
    "Wikipedia (en)": "wikipedia",
    "OpenSubtitles": "opensubtitles",
    "OpenWebText2": "openwebtext2",
    "Gutenberg (PG-19)": "gutenberg",
    "DM Mathematics": "dm-mathematics",
    "Enron Emails": "enron",
    "Books3": "bibliotik",
    "PubMed Abstracts": "pubmed-abstracts",
    "YoutubeSubtitles": "youtubesubtitles",
    "HackerNews": "hackernews",
    "Pile-CC": "commoncrawl",
    "EuroParl": "europarl",
    "USPTO Backgrounds": "uspto",
    "FreeLaw": "freelaw",
    "NIH ExPorter": "nih-exporter",
    "StackExchange": "stackexchange",
    "PubMed Central": "pubmed-central",
    "Ubuntu IRC": "ubuntu-irc",
    "BookCorpus2": "bookcorpus",
    "PhilPapers": "philpapers",
}

# These datasets were too small (in number of docs) to split 10-ways
DATASETS_WITHOUT_SPLIT = [
    "ubuntu-irc",
    "bookcorpus",
    "philpapers",
]

DATASET_COLS = sorted(DATASET_NAMES_DICT.keys())
COLS = ['Model', 'steps'] + DATASET_COLS


def main():
    args = parse_args()
    model = models.create_model(args.model_config_path)
    run = wandb.init(**{'project': args.wandb_project, 'name': f'{model.model_name}_Pile_Test_Metrics'})
    rows = [model.model_name, args.step]

    for name in DATASET_COLS:
        short_name = DATASET_NAMES_DICT[name]
        dataset_indices_path = os.path.join(args.doc_indices_path, short_name, 'group0.json')
        if os.path.exists(dataset_indices_path):
            indices = utils.read_json(dataset_indices_path)
            data_path = args.data_path
        else:
            indices = None
            data_path = os.path.join(args.doc_indices_path, short_name, f'{short_name}.jsonl')

        perplexity_data = compute_perplexity_data(
            model=model,
            data_path=data_path,
            indices=indices,
        )

        aggregate_logprobs = np.concatenate(perplexity_data["all_logprobs"])
        perplexity = float(np.exp(-aggregate_logprobs.mean()))
        utf8_conversion_scalar = perplexity_data['aggregate_length'] / perplexity_data['aggregate_utf8_length']
        bpb = float(np.log2(perplexity) * utf8_conversion_scalar)
        rows.append(bpb)
        print(f'Name:{name} Short Name: {short_name} PPL: {perplexity} BPB: {bpb}')

    my_table = wandb.Table(columns=COLS, data=[rows])
    run.log({"Test Metrics": my_table})
    wandb.run.finish()


if __name__ == "__main__":
    main()
