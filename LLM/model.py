import logging
import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from pynvml import *  # TODO VH: what do we need here? Oplatek asks
from typing import Any, Dict
import os
import openai

openai.api_key = os.environ.get("OPENAI_API_KEY", None)

from prompt import Prompt


def get_model(prompt: Prompt, args):
    model_name = args.model_name
    cache_dir = args.cache_dir
    logging.info(f'Loading model "{model_name}"')

    if model_name.startswith("text-"):
        model = OpenAILLM(None, None, prompt, type=model_name)
    elif model_name.startswith("gpt-"):
        model = OpenAIChatLLM(None, None, prompt, type=model_name)
    elif model_name.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model = HFLLM(model, tokenizer, prompt, type="causal")
    elif model_name.startswith("tiiuae/falcon-"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = HFLLM(model, tokenizer, prompt, type="causal")

    elif model_name.startswith("meta-llama/Llama"):
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        if args.quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=args.use_4bit,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=args.use_nested_quant,
            )
        else:
            bnb_config = None

        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            quantization_config=bnb_config, 
            device_map="auto", 
            use_auth_token=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = HFLLM(model, tokenizer, prompt, type="causal")
    elif any([n in model_name for n in ["opt", "NeoXT"]]):
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            device_map="auto",
            load_in_8bit=True,
        )
        model = HFLLM(model, tokenizer, prompt, type="causal")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            device_map="auto",
            load_in_8bit=True,
        )
        model = HFLLM(model, tokenizer, prompt, type="seq2seq")
    return model


class PromptedLLM:
    def __init__(self, model, tokenizer, prompt: Prompt, type="seq2seq"):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.type = type

    def __call__(self, dialogue_context: str, response: str, **kwargs: Any):
        prompt_str = self.prompt(dialogue_context, response)

        # hack to force the model to generate the qualities: only for multiple qualities
        if len(self.prompt.qualities) > 1:
            forced_words = [f"{w}:" for w in self.prompt.qualities]
        # hacky connection how to force single output
        elif "single_digit" in self.prompt.name:
            forced_words = "digits"
        else:
            forced_words = None

        raw_reply, hidden_states = self._predict(prompt_str, forced_words=forced_words)
        # TODO discussed at https://github.com/oplatek/semetric/pull/66/files#r1155608443 or on slack
        # Vojta added / Oplatek commented out
        # raw_reply = raw_reply.split("\n")[-1]
        metrics, errors = self.prompt.parse_reply(raw_reply)
        if sum(errors.values()) > 0:
            logging.warning(
                f"Errors in reply:\n\t{errors}\n\t{raw_reply=}\n\t{prompt_str=}\n\n"
            )
        logs = {
            "raw_reply": raw_reply,
            "prompt": prompt_str,
            "dialogue_context": dialogue_context,
            "response": response,
        }
        return metrics, errors, logs, hidden_states

    def _predict(self, text):
        raise NotImplementedError("Override this method")


class HFLLM(PromptedLLM):
    def _predict(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(
            self.model.device
        )

    def _predict(self, text, forced_words=None, **kwargs):
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(
            self.model.device
        )

        min_new_tokens, max_new_tokens = None, 100
        forced_decoder_ids, force_words_ids = None, None
        forced_words = forced_words or []
        num_beams = 1
        if len(forced_words) == 0:
            forced_decoder_ids, force_words_ids = None, None
        elif forced_words == "digits":
            # does not work exactly based on docs
            # https://huggingface.co/docs/transformers/main_classes/text_generation
            # force_words_ids = self.tokenizer(
            #     ["1", "2", "3", "4", "5"],
            #     add_prefix_space=True,
            #     add_special_tokens=False,
            # ).input_ids
            force_words_ids = [
                self.tokenizer.encode(dgt) for dgt in ["1", "2", "3", "4", "5"]
            ]

            assert all([len(arr) == 3 for arr in force_words_ids]), force_words_ids
            # TODO what these tokens are
            # [[1, 29871, 29896], [1, 29871, 29906], [1, 29871, 29941], [1, 29871, 29946], [1, 29871, 29945]]
            min_new_tokens = 3
            max_new_tokens = 3

            # From Doc: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
            # force_words_ids(List[List[int]] or List[List[List[int]]], optional)
            #  List of token ids that must be generated. If given a List[List[int]],
            #  this is treated as a simple list of words that must be included, the opposite to bad_words_ids.
            #  If given List[List[List[int]]], this triggers a disjunctive constraint, 
            #  where one can allow different forms of each word.
            force_words_ids = [[arr] for arr in force_words_ids]  # disjunktive digits
            num_beams = 2
            forced_decoder_ids = None

        else:
            force_words_ids = None
            forced_decoder_ids = [
                [n + 1, tid]
                for n, tid in enumerate(self.tokenizer.encode("Qualities: ")[:-1])
            ]
            token_index = len(forced_decoder_ids)
            for fw in forced_words:
                fw_token_ids = self.tokenizer.encode(fw)[:-1]  # ignore '1' in the end
                for fwtid in fw_token_ids:
                    forced_decoder_ids.append([token_index, fwtid])
                    token_index += 1
                token_index += 2

        output = self.model.generate(
            input_ids,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=0.9,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_hidden_states=True,
            force_words_ids = force_words_ids,
            forced_decoder_ids=forced_decoder_ids,
            temperature=0.01,
            num_beams=num_beams,
        )
        generated_sequence = output.sequences
        # extract only last layer
        if self.type == "causal":
            last_layer_hidden_states = [
                hs[-1] for hs in output.hidden_states
            ]  # length = num_gen_tokens
            generated_sequence = generated_sequence[0, input_ids.shape[1] :]
        else:
            last_layer_hidden_states = [
                hs[-1] for hs in output.decoder_hidden_states
            ]  # length = num_gen_tokens
            generated_sequence = generated_sequence[0]

        # extract the relevant generation step for each generated token
        # each element is of shape: (num_return_sequences*batch_size, generated_length??/actually seems always 1 which makes sense/, hidden_size)
        last_layer_hidden_states = [
            hs[0, 0, :] for i, hs in enumerate(last_layer_hidden_states)
        ]
        generated_prediction = self.tokenizer.decode(
            generated_sequence, skip_special_tokens=True
        )
        return generated_prediction, last_layer_hidden_states


class OpenAILLM(PromptedLLM):
    def _predict(self, text, **kwargs):
        completion = openai.Completion.create(
            model=self.type,
            prompt=text,
            temperature=0,
        )
        return completion.choices[0].text, None


class OpenAIChatLLM(PromptedLLM):
    def _predict(self, text, **kwargs):
        try:
            completion = openai.ChatCompletion.create(
                model=self.type,
                messages=[{"role": "user", "content": text}],
                temperature=0,
            )
            return completion.choices[0].message["content"], None
        except:
            return "3.0", None
