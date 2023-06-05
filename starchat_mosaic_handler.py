# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Any, Dict

import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig)


class StarChatModelHandler():
    END_TOKEN = "<|end|>"
    DEFAULT_GENERATE_KWARGS = {
        'max_length': 256,
        'use_cache': True,
        'do_sample': True,
        'top_p': 0.95,
        'temperature': 0.8,
    }

    INPUT_STRINGS_KEY = 'input_strings'

    def __init__(self, model_name: str):
        self.device = torch.cuda.current_device()
        self.model_name = model_name

        config = AutoConfig.from_pretrained(self.model_name,
                                            trust_remote_code=True)
        # config.attn_config['attn_impl'] = 'triton'

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, load_in_8bit=True, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
        )

        # model = AutoModelForCausalLM.from_pretrained(self.model_name,
        #                                              config=config,
        #                                              torch_dtype=torch.bfloat16,
        #                                              trust_remote_code=True)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, config=config, trust_remote_code=True)
        self.tokenizer = tokenizer

        self.generation_config = GenerationConfig(
            temperature=0.2,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids(self.END_TOKEN),
            min_new_tokens=32,
            max_new_tokens=256,
        )

        model = model.eval()
        self.generator = pipeline(task='text-generation',
                                  model=model,
                                  tokenizer=tokenizer,
                                  device=self.device)

    def _parse_inputs(self, inputs: Dict[str, Any]):
        if 'input_strings' not in inputs:
            raise RuntimeError(
                'Input strings must be provided as a list to generate call')

        generate_input = inputs[self.INPUT_STRINGS_KEY]

        # Set default generate kwargs
        generate_kwargs = copy.deepcopy(self.generation_config)
        generate_kwargs['eos_token_id'] = self.tokenizer.eos_token_id

        # If request contains any additional kwargs, add them to generate_kwargs
        for k, v in inputs.items():
            if k not in [self.INPUT_STRINGS_KEY]:
                generate_kwargs[k] = v

        return generate_input, generate_kwargs

    def predict(self, **inputs: Dict[str, Any]):
        generate_input, generate_kwargs = self._parse_inputs(inputs)
        outputs = self.generator(generate_input, **generate_kwargs)
        return [output[0]['generated_text'] for output in outputs]

