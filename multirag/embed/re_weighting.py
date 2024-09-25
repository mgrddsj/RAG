import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"

# os.environ["CUDA_VISIBLE_DEVICES"] = "5,7"
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import torch
from functools import partial
from tqdm import tqdm
import numpy as np
from .prompt import get_prompt

RAG_prompt1 = """Given the following information: \n"""
RAG_prompt2 = """Answer the following question based on the given information or your internal knowledge with one or few words without the source.
Question: {question}
Answer: {answer}"""
# RAG_prompt2 = """Answer the following question based on the given information or your internal knowledge.
# Question: {question}
# Answer: {answer}"""


class Re_Weighting_Strategy:

    def __init__(self, model_name: str = "Llama-2-13b-chat-hf", layers_to_be_modified: dict = dict(), bad_words_ids=[]):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.model_name = model_name
        self.bad_words_ids = bad_words_ids
        self.num_hidden_layers = self.model.config.num_hidden_layers
        self.model_num_attention_heads = self.model.config.num_attention_heads
        print(f"Model {model_name} loaded successfully.")
        print(f"Number of hidden layers: {self.num_hidden_layers}")
        print(f"Number of attention heads: {self.model_num_attention_heads}")
        if not layers_to_be_modified:
            layers_to_be_modified = {i: list(range(self.model_num_attention_heads)) for i in range(self.num_hidden_layers)}
        self.layers_to_be_modified = layers_to_be_modified

    def edit_attention_mask(self, module: torch.nn.Module, input_args: tuple, input_kwargs: dict, attention_weight: list, head_idx: list = []):
        weight_len = attention_weight.size()[-1]
        dtype, device = input_kwargs['hidden_states'].dtype, input_kwargs['hidden_states'].device
        if input_kwargs['attention_mask'] == None:
            bsz, head_dim = 1, 1
            tgt_len = input_kwargs['hidden_states'].size()[1]
            src_len = input_kwargs['position_ids'][0][-1] + 1
            if tgt_len == 1:
                attention_mask = torch.zeros([bsz, head_dim, tgt_len, src_len], dtype=dtype, device=device)
            else:
                min_value = torch.finfo(dtype).min
                upper_triangle_matrix = torch.triu(torch.full((tgt_len, src_len), min_value, dtype=dtype, device=device), diagonal=1)
                attention_mask = upper_triangle_matrix.unsqueeze(0).unsqueeze(0).expand(bsz, head_dim, tgt_len, src_len)
        else:
            attention_mask = input_kwargs['attention_mask'].clone()
        bsz, head_dim, tgt_len, src_len = attention_mask.size()
        if head_dim == 1:
            attention_mask = attention_mask.repeat(1, self.model_num_attention_heads, 1, 1)
            head_dim = self.model_num_attention_heads
        # dtype, device = attention_mask.dtype, attention_mask.device
        expanded_weight = attention_weight.unsqueeze(0).unsqueeze(0).repeat(bsz, head_dim, tgt_len, 1).to(dtype=dtype, device=device)
        mask = (attention_mask[..., :weight_len] == 0.0)
        for h in head_idx:
            attention_mask[:, h, :, :weight_len][mask[:, h, :, :]] = expanded_weight[:, h, :, :][mask[:, h, :, :]]
        input_kwargs['attention_mask'] = attention_mask
        return input_args, input_kwargs

    def decode_with_special_attention(self, question: str = '', paras: list = [], scores: list = [], answer: str = ''):
        add_special_tokens = True
        if self.model_name.find("Llama-3") != -1:
            prompt = get_prompt(context=paras, question=question, answer='', type='with_contexts')
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant!"
                },
                {
                    "role": "user",
                    "content": f"{prompt}"
                },
            ]
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            if answer != '':
                prompt += answer
        elif self.model_name.find('Qwen') != -1:
            prompt = get_prompt(context=paras, question=question, answer='', type='with_contexts')
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": f"{prompt}"
                },
            ]
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            add_special_tokens = False
            if answer != '':
                prompt += answer
        else:
            prompt = get_prompt(context=paras, question=question, answer=answer, type='with_contexts')
        model_inputs = self.tokenizer([prompt], return_tensors="pt", return_offsets_mapping=True, add_special_tokens=add_special_tokens).to("cuda")
        attention_weight = model_inputs['attention_mask'].clone().to(torch.float16)
        for i, p in enumerate(paras):
            para = ("Passage-%d: " % i) + p + '\n'
            start_idx = prompt.find(para)
            end_idx = start_idx + len(para) - 1
            start_id_pos = None
            end_id_pos = None
            for idx, x in enumerate(model_inputs['offset_mapping'][0]):
                if start_idx >= x[0]:
                    start_id_pos = idx
                if end_idx >= x[0]:
                    end_id_pos = idx
            attention_weight[:, start_id_pos:end_id_pos + 1] = torch.full((1, end_id_pos + 1 - start_id_pos), scores[i]).to("cuda").to(torch.float16)
        model_inputs.pop('offset_mapping')

        attention_heads = self.get_attention_heads_for_question(question=question)
        return model_inputs, attention_weight, attention_heads
    
    def get_attention_heads_for_question(self, question: str = ''):
        # TODO: To be implemented with CrAM
        model_inputs = self.tokenizer([question], return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False).to("cuda")
        attention_weight = model_inputs['attention_mask'].clone().to(torch.float16)
        return attention_weight

    @torch.no_grad()
    def run_RAG_with_attention_weighting(self, question: str = '', paras: list = [], scores: list = []):
        model_inputs, attention_weight, attention_heads = self.decode_with_special_attention(question=question, paras=paras, scores=scores)
        registered_hooks = []

        for layer_idx, head_idx in self.layers_to_be_modified.items():
            module = self.model.get_submodule(f"model.layers.{layer_idx}.self_attn")
            hook_func = partial(self.edit_attention_mask, attention_weight=torch.log(attention_weight), head_idx=head_idx)
            registered_hook = module.register_forward_pre_hook(hook_func, with_kwargs=True)
            registered_hooks.append(registered_hook)

        prompt = self.tokenizer.decode(model_inputs['input_ids'][0][1:])
        para_dict = {"do_sample": False, "max_new_tokens": 100}
        if self.bad_words_ids:
            para_dict["bad_words_ids"] = self.bad_words_ids
        if self.model_name.find("Llama-3") != -1:
            para_dict["eos_token_id"] = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        outputs = self.model.generate(**model_inputs, **para_dict)
        output = self.tokenizer.decode(outputs[0][1:-1])
        prompt_end_index = output.find(prompt) + len(prompt)
        output = output[prompt_end_index:]
        for registered_hook in registered_hooks:
            registered_hook.remove()
        return prompt, output


class Find_Best_Heads(Re_Weighting_Strategy):

    def __init__(self, model_name: str = "Llama-2-13b-chat-hf", layers_to_be_modified: list = []):
        super().__init__(model_name=model_name)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.model_num_hidden_layers = self.model.config.num_hidden_layers
        # self.model_num_hidden_layers = 2
        self.model_num_attention_heads = self.model.config.num_attention_heads
        self.hidden_size = self.model.config.hidden_size

    @torch.no_grad()
    def cal_logits(self, question: str = '', paras: list = [], scores: list = [], right_answer: str = '', wrong_answer: str = 'my name is'):
        answer = wrong_answer
        answer_ids = self.tokenizer([answer], return_tensors="pt")['input_ids'][0, 1:]
        model_inputs, attention_weight, attention_heads = self.decode_with_special_attention(question=question, paras=paras, scores=scores, answer=answer)
        registered_hooks = []

        self.ori_logits = self.model(**model_inputs, return_dict=True)['logits'].clone()
        self.ori_prob_sum = self.ori_logits[0, -len(answer_ids):][np.arange(len(answer_ids)), answer_ids].sum()
        prob_change = []
        for layer_idx in tqdm(range(self.model_num_hidden_layers)):
            module = self.model.get_submodule(f"model.layers.{layer_idx}.self_attn")
            prob_change_layer = []
            for head_idx in range(self.model_num_attention_heads):
                hook_func = partial(self.edit_attention_mask, attention_weight=torch.log(attention_weight), head_idx=[head_idx])
                registered_hook = module.register_forward_pre_hook(hook_func, with_kwargs=True)
                registered_hooks.append(registered_hook)
                current_logits = self.model(**model_inputs, return_dict=True)['logits'].clone()
                current_prob_sum = current_logits[0, -len(answer_ids):][np.arange(len(answer_ids)), answer_ids].sum()
                prob_change_layer.append((self.ori_prob_sum - current_prob_sum).item())
                for registered_hook in registered_hooks:
                    registered_hook.remove()
            prob_change.append(prob_change_layer)
        return prob_change
