from __future__ import annotations

import ctypes
from collections.abc import Iterable
import os
from typing import Any, Protocol, Optional, List, Tuple

from huggingface_hub.inference._text_generation import TextGenerationStreamResponse, Token
import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList

from conversation import Conversation

TOOL_PROMPT = 'Answer the following questions as best as you can. You have access to the following tools:'

MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
PT_PATH = os.environ.get('PT_PATH', None)
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)
QUANTIZE_BIT = os.environ.get("QUANTIZE_BIT", 8)
USE_FASTLLM = os.environ.get("USE_FASTLLM", True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FLLM_DIR_PATH = 'fllm'

TOKEN_DICT = {
            'system': 64794,
            'user': 64795,
            'assistant': 64796,
            'abservation': 64797
}

try:
    from fastllm_pytools import llm
except:
    USE_FASTLLM = False

# for Mac Computer like M1
# You Need Use Pytorch compiled with Metal
# DEVICE = 'mps'

# for AMD gpu likes MI100 (Not Official Steady Support yet)
# You Need Use Pytorch compiled with ROCm
# DEVICE = 'cuda'

# for Intel gpu likes A770 (Not Official Steady Support yet)
# You Need Use Pytorch compiled with oneDNN and install intel-extension-for-pytorch
# import intel_extension_for_pytorch as ipex
# DEVICE = 'xpu'

# for Moore Threads gpu like MTT S80 (Not Official Steady Support yet)
# You Need Use Pytorch compiled with Musa
# DEVICE = 'musa'


@st.cache_resource
def get_client() -> Client:
    client = HFClient(model_path=MODEL_PATH, 
                      tokenizer_path=TOKENIZER_PATH, 
                      pt_checkpoint=PT_PATH, 
                      quantize_bit=QUANTIZE_BIT,
                      use_fastllm=USE_FASTLLM,
                      DEVICE=DEVICE)
    return client


class Client(Protocol):
    def generate_stream(self,
                        system: Optional[str],
                        tools: Optional[List[dict]],
                        history: List[Conversation],
                        **parameters: Any
                        ) -> Iterable[TextGenerationStreamResponse]:
        ...


def stream_chat(self, 
                tokenizer, 
                query: str, 
                history: Optional[List[dict]]=None, 
                role: str="user",
                max_length: int=8192, 
                do_sample: bool=True,
                top_p: float=0.8,
                top_k: int=1,
                temperature: float=0.8,
                repetition_penalty: float=1.0,
                length_penalty: int=1.0, 
                num_beams: int=1,
                logits_processor: Optional[LogitsProcessorList]=None,
                return_past_key_values: bool=False, 
                **kwargs):

    class InvalidScoreLogitsProcessor(LogitsProcessor):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                scores.zero_()
                scores[..., 5] = 5e4
            return scores

    if history is None:
        history = []
    if logits_processor is None:
        logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    eos_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"),
                    tokenizer.get_command("<|observation|>")]
    gen_kwargs = {"max_length": max_length,
                  "do_sample": do_sample,
                  "top_p": top_p,
                  "temperature": temperature,
                  "logits_processor": logits_processor,
                  "repetition_penalty": repetition_penalty,
                  "length_penalty": length_penalty,
                  "num_beams": num_beams,
                  **kwargs
                  }

    print(gen_kwargs)
    if past_key_values is None:
        inputs = tokenizer.build_chat_input(query, history=history, role=role)
    else:
        inputs = tokenizer.build_chat_input(query, role=role)
    inputs = inputs.to(self.device)
    if past_key_values is not None:
        past_length = past_key_values[0][0].shape[0]
        if self.transformer.pre_seq_len is not None:
            past_length -= self.transformer.pre_seq_len
        inputs.position_ids += past_length
        attention_mask = inputs.attention_mask
        attention_mask = torch.cat((attention_mask.new_ones(1, past_length), attention_mask), dim=1)
        inputs['attention_mask'] = attention_mask
    history.append({"role": role, "content": query})
    print("input_shape>", inputs['input_ids'].shape)

    input_sequence_length = inputs['input_ids'].shape[1]

    if max_length < input_sequence_length <= self.config.seq_length:
        yield "Current input sequence length {} exceeds sequence length set in generation parameters {}. The maximum model sequence length is {}. You may adjust the generation parameter to enable longer chat history.".format(
            input_sequence_length, max_length, self.config.seq_length
        ), history
        return

    if input_sequence_length > self.config.seq_length:
        yield "Current input sequence length {} exceeds maximum model sequence length {}. Unable to generate tokens.".format(
            input_sequence_length, self.config.seq_length
        ), history
        return

    for outputs in self.stream_generate(**inputs, past_key_values=past_key_values,
                                        eos_token_id=eos_token_id, return_past_key_values=return_past_key_values,
                                        **gen_kwargs):
        if return_past_key_values:
            outputs, past_key_values = outputs
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        if response and response[-1] != "�":
            new_history = history
            if return_past_key_values:
                yield response, new_history, past_key_values
            else:
                yield response, new_history


def stream_chat_faster(faster_model: llm.model,
                       query: str,
                       history: Optional[List[dict]],
                       role: str="user",
                       system: str = '',
                       max_length: int=8192, 
                       do_sample: bool=True,
                       top_p: float=0.8,
                       top_k: int=1,
                       temperature: float=0.8,
                       repetition_penalty: float=1.0,
                       one_by_one: bool=True,
                       stop_token_ids: Optional[List[int]]=None,
                       **kwargs,
                       ):
    def get_prompt(query, history, role, system):
        
        prompt = f"<FLM_FIX_TOKEN_{TOKEN_DICT['system']}>\n{system}\n"
        for old_query, response in history:
            prompt += f"<FLM_FIX_TOKEN_{TOKEN_DICT['user']}>\n{old_query}\n<FLM_FIX_TOKEN_{TOKEN_DICT['assistant']}>\n{response}\n"
        prompt += f"<FLM_FIX_TOKEN_{TOKEN_DICT['user']}>\n{query}\n<FLM_FIX_TOKEN_{TOKEN_DICT['assistant']}>\n"
        # print(prompt) # test
        return prompt
    
    history_new: List[Tuple[str, str]] = []
    for i, chat_record in enumerate(history):
        if chat_record['role'] == 'user':
            if i + 1 < len(history) and history[i+1]['role'] == 'assistant':
                history_new.append((history[i]['content'], history[i+1]['content']))

    prompt = get_prompt(query, history_new, role, system)
    stop_token_len, stop_token_list = faster_model.stop_token_ctypes(stop_token_ids)
    handle = llm.fastllm_lib.launch_response_str_llm_model(faster_model.model, 
                                                           prompt.encode(),
                                                           ctypes.c_int(max_length), 
                                                           ctypes.c_bool(do_sample),
                                                           ctypes.c_float(top_p),
                                                           ctypes.c_int(top_k),
                                                           ctypes.c_float(temperature),
                                                           ctypes.c_float(repetition_penalty),
                                                           ctypes.c_bool(False),
                                                           stop_token_len,
                                                           stop_token_list)
    
    res = ""
    ret = b''
    fail_cnt = 0
    while True:
        ret += llm.fastllm_lib.fetch_response_str_llm_model(faster_model.model, handle)
        cur = ""
        try:
            cur = ret.decode()
            ret = b''
        except:
            fail_cnt += 1
            if (fail_cnt == 20):
                break
            else:
                continue
        fail_cnt = 0
        if (cur == "<flmeos>"):
            break
        if one_by_one:
            yield cur
        else:
            res += cur
            yield res


class HFClient(Client):
    def __init__(self, 
                 model_path: str, 
                 tokenizer_path: str, 
                 pt_checkpoint: Optional[str]=None,
                 quantize_bit: Optional[int]=None,
                 use_fastllm: Optional[bool]=None,
                 DEVICE = 'cpu'):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.use_fastllm = use_fastllm

        if use_fastllm:
            assert 'llm' in globals(), 'FastLLM is not available.'
            print('FastLLM is available.')
            
            if self._has_fllm_model(model_path, quantize_bit):
                print('Found existed fllm model.')
                self.faster_model = self._load_fllm_model(model_path, quantize_bit)
                return

            self.faster_model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            if quantize_bit is not None:
                assert quantize_bit in [4, 8], 'Only support 4-bit and 8-bit quantize.'
                self.faster_model = llm.from_hf(self.faster_model, self.tokenizer, dtype='int{}'.format(quantize_bit))
                print('Use {}-bit quantized model.'.format(quantize_bit))
            else:
                self.faster_model = llm.from_hf(self.faster_model, self.tokenizer, dtype='float16')
            self._save_fllm_model(self.faster_model, model_path, quantize_bit)
            return

        if pt_checkpoint is not None:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, pre_seq_len=128)
            if quantize_bit is not None:
                assert quantize_bit in [4, 8], 'Only support 4-bit and 8-bit quantize.'
                self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, config=config).quantize(quantize_bit)
                print('Use {}-bit quantized model.'.format(quantize_bit))
            else:
                self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, config=config)
            prefix_state_dict = torch.load(os.path.join(pt_checkpoint, "pytorch_model.bin"))
            new_prefix_state_dict = {}
            for k, v in prefix_state_dict.items():
                if k.startswith("transformer.prefix_encoder."):
                    new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
            print("Loaded from pt checkpoints", new_prefix_state_dict.keys())
            self.model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

        else:
            if quantize_bit is not None:
                assert quantize_bit in [4, 8], 'Only support 4-bit and 8-bit quantize.'
                self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).quantize(quantize_bit)
                print('Use {}-bit quantized model.'.format(quantize_bit))
            else:
                self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

        self.model = self.model.to(DEVICE).eval() if 'cuda' in DEVICE else self.model.float().to(DEVICE).eval()


    def _calculate_md5(self, input_string: str):
        from hashlib import md5
        md5_hash = md5()
        md5_hash.update(input_string.encode('utf-8'))
        return md5_hash.hexdigest()
    

    def _has_fllm_model(self,
                       model_path: str,
                       quantize_bit: Optional[int]):
        md5_result = self._calculate_md5(f'{model_path}_{quantize_bit}')
        if os.path.exists(os.path.join(FLLM_DIR_PATH, f'{md5_result}.fllm')):
            return True
        return False
    
    
    def _save_fllm_model(self,
                        fllm_model,
                        model_path: str,
                        quantize_bit: Optional[int]):
        md5_result = self._calculate_md5(f'{model_path}_{quantize_bit}')
        fllm_model_path = os.path.join(FLLM_DIR_PATH, f'{md5_result}.fllm')
        fllm_model.save(fllm_model_path)


    def _load_fllm_model(self,
                       model_path: str,
                       quantize_bit: Optional[int]):
        md5_result = self._calculate_md5(f'{model_path}_{quantize_bit}')
        fllm_model_path = os.path.join(FLLM_DIR_PATH, f'{md5_result}.fllm')
        model = llm.model(fllm_model_path)
        return model


    def generate_stream(self,
                        system: Optional[str],
                        tools: Optional[List[dict]],
                        history: List[Conversation],
                        **parameters: Any
                        ) -> Iterable[TextGenerationStreamResponse]:
        chat_history = [{
            'role': 'system',
            'content': system if not tools else TOOL_PROMPT,
        }]

        def remove_surrounding_chars(s: str, start: Optional[str]=None, end: Optional[str]=None) -> str:
            if start and s.startswith(start):
                s = s[len(start):]
            if end and s.endswith(end):
                s = s[:-len(end)]
            return s

        if tools:
            chat_history[0]['tools'] = tools

        for conversation in history[:-1]:
            chat_history.append({
                'role': remove_surrounding_chars(str(conversation.role), '<|', '|>'),
                'content': conversation.content,
            })

        query = history[-1].content
        role = remove_surrounding_chars(str(history[-1].role), '<|', '|>')

        text = ''

        if not self.use_fastllm:
            for new_text, _ in stream_chat(self.model,
                                           self.tokenizer,
                                           query,
                                           chat_history,
                                           role,
                                           **parameters,
                                           ):
                word = remove_surrounding_chars(new_text, text)
                word_stripped = word.strip()
                text = new_text
                yield TextGenerationStreamResponse(
                    generated_text=text,
                    token=Token(
                        id=0,
                        logprob=0,
                        text=word,
                        special=word_stripped.startswith('<|') and word_stripped.endswith('|>'),
                    )
                )
        else:
            generated_text = ''
            for new_generate_text in stream_chat_faster(self.faster_model,
                                               query,
                                               chat_history,
                                               role,
                                               system,
                                               **parameters,
                                               ):
                new_generate_text_stripped = new_generate_text.strip()
                yield TextGenerationStreamResponse(
                    generated_text=generated_text,
                    token=Token(
                        id=0,
                        logprob=0,
                        text=new_generate_text,
                        special=new_generate_text_stripped.startswith('<|') and \
                            new_generate_text_stripped.endswith('|>'),
                    )
                )
                generated_text += new_generate_text

