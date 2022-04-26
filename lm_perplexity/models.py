import numpy as np
import os
from typing import Optional

import torch
import torch.nn as nn
import einops
import lm_perplexity.utils as utils
from torch.nn import functional as F

class LM:
    def get_perplexity_data(self, text) -> Optional[dict]:
        raise NotImplementedError

    @classmethod
    def create_from_config(cls, path):
        raise NotImplementedError


class AppliedML42LM(LM):
    def __init__(self,
                 model_name,
                 tokenizer_path,
                 model_ckpt_path,
                 max_seq_len,
                 context_len,
                 device,
                 verbose=False):
        from language_modeling.models import GPT
        import sentencepiece as spm

        self.tokenizer: spm.SentencePieceProcessor
        self.model: GPT

        self.model_name = model_name
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
        self.vocab_size = self.tokenizer.vocab_size()

        self.model = GPT.load_from_checkpoint(model_ckpt_path)
        self.model.to(device)
        self.model.eval()

        self.max_seq_len = max_seq_len
        self.context_len = context_len

        self.end_of_text_token_id = self.tokenizer.PieceToId('[eod]')

        self.verbose = verbose

    def get_perplexity_data(self, text) -> Optional[dict]:
        input_ids = self.tokenizer.EncodeAsIds(text)
        rolling_token_windows = utils.get_rolling_token_windows(
            token_list=input_ids,
            prefix_token=self.end_of_text_token_id,
            max_seq_len=self.max_seq_len,
            context_len=self.context_len,
        )

        # noinspection PyListCreation
        all_logprobs = []
        all_positions = []

        # Remaining windows
        for input_tokens, pred_tokens in rolling_token_windows:
            block_output = self.get_token_logprobs(
                input_tokens=input_tokens,
                pred_tokens=pred_tokens,
            )
            all_logprobs.append(block_output["logprobs"])
            all_positions.append(block_output["positions"])

        if not all_logprobs:
            return None

        # Gather
        all_logprobs = np.concatenate(all_logprobs)
        all_positions = np.concatenate(all_positions)
        assert len(all_logprobs) == len(input_ids)
        return {
            "logprobs": all_logprobs,
            "positions": all_positions,
            "length": len(all_logprobs),
            "utf8_length": len(text.encode('utf-8')),
        }

    def get_token_logprobs(self, input_tokens, pred_tokens):
        positions = np.arange(len(input_tokens) - len(pred_tokens), len(input_tokens))
        x = input_tokens + [self.tokenizer.pad_id()] * (self.max_seq_len - len(input_tokens))
        x = torch.as_tensor(x, device=self.model.device)
        x = einops.rearrange(x, 'max_seq_len -> 1 max_seq_len', max_seq_len=self.max_seq_len)
        y = torch.as_tensor(pred_tokens, device=self.model.device)

        with torch.no_grad():
            y_hat = einops.rearrange(self.model(x), 'batch seq_len vocab -> (batch seq_len) vocab')
        y_hat = y_hat[positions, :]
        neg_logprobs = F.cross_entropy(y_hat, y, reduction='none').detach().cpu().numpy()

        if self.verbose:
            print("Context:", self.tokenizer.IdToPiece(input_tokens))
            print("Predicting:", self.tokenizer.IdToPiece(pred_tokens))
            print("Perplexity:", np.exp(neg_logprobs.mean()))
            print()

        return {
            "logprobs": - neg_logprobs,
            "positions": positions,
        }

    @classmethod
    def create_from_config(cls, config):
        return cls(**config)



class GPT3LM(LM):

    def __init__(self, engine, context_len=1024, max_seq_len=2048, verbose=False):
        import openai
        import transformers
        self.engine = engine
        self.context_len = context_len
        self.max_seq_len = max_seq_len
        self.wb = utils.WaitBlocker()
        self.verbose = verbose

        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2-xl')
        self.end_of_text_token_id = self.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])[0]

        # Read from environment variable OPENAI_API_SECRET_KEY
        openai.api_key = os.environ["OPENAI_API_SECRET_KEY"]

    # noinspection DuplicatedCode
    def get_perplexity_data(self, text) -> Optional[dict]:
        input_ids = self.tokenizer.encode_plus(text)["input_ids"]
        rolling_token_windows = utils.get_rolling_token_windows(
            token_list=input_ids,
            prefix_token=self.end_of_text_token_id,
            max_seq_len=self.max_seq_len,
            context_len=self.context_len,
        )

        # noinspection PyListCreation
        all_logprobs = []
        all_positions = []

        # Remaining windows
        for input_tokens, pred_tokens in rolling_token_windows:
            block_output = self.get_token_logprobs(
                input_tokens=input_tokens,
                pred_tokens=pred_tokens,
            )
            all_logprobs.append(block_output["logprobs"])
            all_positions.append(block_output["positions"])

        if not all_logprobs:
            return None

        # Gather
        all_logprobs = np.concatenate(all_logprobs)
        all_positions = np.concatenate(all_positions)
        assert len(all_logprobs) == len(input_ids)
        return {
            "logprobs": all_logprobs,
            "positions": all_positions,
            "length": len(all_logprobs),
            "utf8_length": len(text.encode('utf-8')),
        }

    def get_token_logprobs(self, input_tokens, pred_tokens):
        pred_start = len(input_tokens) - len(pred_tokens) + 1
        # We're going to stitch together the input_tokens and pred_tokens
        # In the longest case, this gets us to length = max_seq_len+1 (which the API works with)
        assert input_tokens[pred_start:] == pred_tokens[:-1]
        token_ids = input_tokens + [pred_tokens[-1]]
        with self.wb.check_valid():
            response = openai.Completion.create(
                engine=self.engine,
                prompt=token_ids,
                max_tokens=0,
                temperature=0.0,
                logprobs=0,
                echo=True,
            )
        logprobs = np.array(response["choices"][0]["logprobs"]["token_logprobs"][pred_start:])
        if self.verbose:
            print("Context:", self.tokenizer.convert_ids_to_tokens(token_ids))
            print("Predicting:", self.tokenizer.convert_ids_to_tokens(token_ids)[pred_start:])
            print("Perplexity:", np.exp(-logprobs.mean()))
            print()

        positions = np.arange(pred_start-1, pred_start-1 + len(token_ids[pred_start:]))

        return {
            "logprobs": logprobs,
            "positions": positions,
        }

    @classmethod
    def create_from_config(cls, config):
        return cls(**config)


class GPT2LM(LM):

    def __init__(self, model_name, device="cuda:0", context_len=512, max_seq_len=1024, verbose=False):
        self.model_name = model_name
        self.device = torch.device(device)
        self.context_len = context_len
        self.max_seq_len = max_seq_len
        self.verbose = verbose

        torch.set_grad_enabled(False)
        self.model = transformers.GPT2LMHeadModel.from_pretrained(model_name).eval().to(self.device)
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained(model_name)
        self.end_of_text_token_id = self.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])[0]

    # noinspection DuplicatedCode
    def get_perplexity_data(self, text) -> Optional[dict]:
        input_ids = self.tokenizer.encode_plus(text)["input_ids"]
        rolling_token_windows = utils.get_rolling_token_windows(
            token_list=input_ids,
            prefix_token=self.end_of_text_token_id,
            max_seq_len=self.max_seq_len,
            context_len=self.context_len,
        )

        # noinspection PyListCreation
        all_logprobs = []
        all_positions = []

        # Remaining windows
        for input_tokens, pred_tokens in rolling_token_windows:
            block_output = self.get_token_logprobs(
                input_tokens=input_tokens,
                pred_tokens=pred_tokens,
            )
            all_logprobs.append(block_output["logprobs"])
            all_positions.append(block_output["positions"])

        if not all_logprobs:
            return None

        # Gather
        all_logprobs = np.concatenate(all_logprobs)
        all_positions = np.concatenate(all_positions)
        assert len(all_logprobs) == len(input_ids)
        return {
            "logprobs": all_logprobs,
            "positions": all_positions,
            "length": len(all_logprobs),
            "utf8_length": len(text.encode('utf-8')),
        }

    def get_token_logprobs(self, input_tokens, pred_tokens):
        input_tokens = torch.tensor(input_tokens).long().to(self.device)
        pred_tokens = torch.tensor(pred_tokens).long().to(self.device)
        output = self.model(input_tokens, return_dict=True)
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        neg_logprobs = loss_fct(
            output.logits[-len(pred_tokens):],
            pred_tokens,
        ).detach().cpu().numpy()
        if self.verbose:
            print("Context:", self.tokenizer.convert_ids_to_tokens(input_tokens))
            print("Predicting:", self.tokenizer.convert_ids_to_tokens(pred_tokens))
            print("Perplexity:", np.exp(neg_logprobs.mean()))
            print()

        positions = np.arange(len(input_tokens) - len(pred_tokens), len(input_tokens))

        return {
            "logprobs": - neg_logprobs,
            "positions": positions,
        }

    @classmethod
    def create_from_config(cls, config):
        return cls(**config)


def create_model(json_path):
    config = utils.read_json(json_path)
    model_type = config.pop("model_type")
    if model_type == "gpt3":
        model = GPT3LM.create_from_config(config)
    elif model_type == "gpt2":
        model = GPT2LM.create_from_config(config)
    elif model_type == "appliedml42LM":
        model = AppliedML42LM.create_from_config(config)
    else:
        raise KeyError(model_type)
    return model
