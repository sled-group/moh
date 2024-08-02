import torch
import transformers
from torch import nn
from PIL import Image
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForVision2Seq, AutoProcessor, LlavaNextProcessor, LlavaNextForConditionalGeneration


import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image

from minigpt4.common.dist_utils import get_rank
from minigpt4.models import load_preprocess

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from PIL import Image
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn
import json

class BaseModelHandler:
    def __init__(self, model_name, model_size, model_path, device_map):
        self.model_name = model_name
        self.model_size = model_size
        self.processor, self.model = self.init_model(model_path, device_map)

    def init_model(self, model_path):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def generate_response(self, prompt, image):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
# class MiniCPMHandler(BaseModelHandler):
#     def init_model(self, model_path, device_map):
#         tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
#         model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             torch_dtype=torch.float16,
#             low_cpu_mem_usage=True,
#             trust_remote_code=True,
#             device_map=device_map,
#         ).eval()
        
#         return tokenizer, model
    
#     def generate_response(self, prompt, image):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         msgs = [{'role': 'user', 'content': prompt}]
        
#         with torch.no_grad():
#             res = self.model.chat(  # Access the original model using `module`
#                 image=image,
#                 msgs=msgs,
#                 tokenizer=self.processor,
#                 sampling=True,  # if sampling=False, beam_search will be used by default
#                 temperature=0.7,
#                 # system_prompt=''  # pass system_prompt if needed
#             )

#         return res

# class CogVLM2Handler(BaseModelHandler):
#     def init_model(self, model_path, device_map):
#         tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
#         TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
#         self.TORCH_TYPE = TORCH_TYPE
#         model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             torch_dtype=self.TORCH_TYPE,
#             low_cpu_mem_usage=True,
#             trust_remote_code=True,
#             device_map=device_map,
#         ).eval()
        
#         return tokenizer, model
    
#     def generate_response(self, prompt, image):
#         history = []
#         input_by_model = self.model.build_conversation_input_ids(
#             self.processor,
#             query=prompt,
#             history=history,
#             images=[image],
#             template_version='chat'
#         )
#         inputs = {
#             'input_ids': input_by_model['input_ids'].unsqueeze(0).to("cuda"),
#             'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to("cuda"),
#             'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to("cuda"),
#             'images': [[input_by_model['images'][0].to("cuda").to(self.TORCH_TYPE)]] if image is not None else None,
#         }
#         gen_kwargs = {
#             "max_new_tokens": 2048,
#             "pad_token_id": 128002,  
#         }
#         with torch.no_grad():
#             outputs = self.model.generate(**inputs, **gen_kwargs)
#             outputs = outputs[:, inputs['input_ids'].shape[1]:]
#             response = self.processor.decode(outputs[0])
#             response = response.split("<|end_of_text|>")[0]
#         return response

# class YiVLModelHandler(BaseModelHandler):
#     def init_model(self, model_path, device_map):
#         # Monkey patch of LlavaMultiModalProjector is mandatory
#         class LlavaMultiModalProjectorYiVL(nn.Module):
#             def __init__(self, config: "LlavaConfig"):
#                 super().__init__()
#                 self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
#                 self.linear_2 = nn.LayerNorm(config.text_config.hidden_size, bias=True)
#                 self.linear_3 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)
#                 self.linear_4 = nn.LayerNorm(config.text_config.hidden_size, bias=True)
#                 self.act = nn.GELU()

#             def forward(self, image_features):
#                 hidden_states = self.linear_1(image_features)
#                 hidden_states = self.linear_2(hidden_states)
#                 hidden_states = self.act(hidden_states)
#                 hidden_states = self.linear_3(hidden_states)
#                 hidden_states = self.linear_4(hidden_states)
#                 return hidden_states

#         transformers.models.llava.modeling_llava.LlavaMultiModalProjector = LlavaMultiModalProjectorYiVL

#         model = AutoModelForVision2Seq.from_pretrained(
#             model_path, 
#             torch_dtype=torch.float16, 
#             low_cpu_mem_usage=True,
#             device_map=device_map,
#         )
#         processor = AutoProcessor.from_pretrained(model_path)
        
#         return processor, model
    
#     def generate_response(self, prompt, image):
#         prompt += "<image>"
#         messages = [{ "role": "user", "content": prompt }]
#         text_inputs = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
#         inputs = self.processor(text=text_inputs, images=image, return_tensors='pt').to("cuda")

#         output = self.model.generate(**inputs, max_new_tokens=200)
#         decoded_output = self.processor.batch_decode(output, skip_special_tokens=True)
#         return decoded_output[0].split("Assistant:")[-1].strip()

# class LlavaNextHandler(BaseModelHandler):
#     def init_model(self, model_path, device_map):
#         processor = LlavaNextProcessor.from_pretrained(model_path)
#         model = LlavaNextForConditionalGeneration.from_pretrained(
#             model_path,
#             torch_dtype=torch.float16,
#             low_cpu_mem_usage=True,
#             device_map=device_map,
#         )
#         return processor, model
    
#     def generate_response(self, prompt, image):
#         prompt_llava34 = f"<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{prompt}<|im_end|><|im_start|>assistant\n"
#         messages = [{ "role": "user", "content": prompt_llava34}]
#         text_inputs = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
#         inputs = self.processor(text=text_inputs, images=image, return_tensors='pt').to("cuda")

#         output = self.model.generate(**inputs, max_new_tokens=200)
#         decoded_output = self.processor.batch_decode(output, skip_special_tokens=True)
#         response = decoded_output[0].split("assistant\n \n")[-1].strip()
#         return response

class OperaLLaVAHandler(BaseModelHandler):
    def __init__(self, model_name, model_size, model_path, device_map):
        self.model_name = model_name
        self.model_size = model_size
        self.vis_processors, self.txt_processors, self.model = self.init_model(model_path, device_map)
        
    def init_model(self, model_path, device_map):
        MODEL_EVAL_CONFIG_PATH = {
            "minigpt4": "eval_configs/minigpt4_eval.yaml",
            "instructblip": "eval_configs/instructblip_eval.yaml",
            "lrv_instruct": "eval_configs/lrv_instruct_eval.yaml",
            "shikra": "eval_configs/shikra_eval.yaml",
            "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
        }

        POPE_PATH = {
            "random": "coco_pope/coco_pope_random.json",
            "popular": "coco_pope/coco_pope_popular.json",
            "adversarial": "coco_pope/coco_pope_adversarial.json",
        }

        INSTRUCTION_TEMPLATE = {
            "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
            "instructblip": "<ImageHere><question>",
            "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
            "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
            "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:"
        }


        def setup_seeds(config):
            seed = config.run_cfg.seed + get_rank()

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            cudnn.benchmark = False
            cudnn.deterministic = True

        parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
        parser.add_argument("--model", type=str, help="model")
        parser.add_argument("--gpu-id", type=int, help="specify the gpu to load the model.")
        parser.add_argument(
            "--options",
            nargs="+",
            help="override some settings in the used config, the key-value pair "
            "in xxx=yyy format will be merged into config file (deprecate), "
            "change to --cfg-options instead.",
        )
        parser.add_argument("--data_path", type=str, default="rebuttal/data/llava-v1.5-7b/", help="data path")
        parser.add_argument("--batch_size", type=int, help="batch size")
        parser.add_argument("--num_workers", type=int, default=2, help="num workers")
        args = parser.parse_known_args()[0]


        args.model = "llava-1.5"
        args.gpu_id = "0"
        args.batch_size = 1


        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
        cfg = Config(args)
        setup_seeds(cfg)
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        # ========================================
        #             Model Initialization
        # ========================================
        print('Initializing Model')

        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to(device)
        model.eval()
        processor_cfg = cfg.get_config().preprocess
        processor_cfg.vis_processor.eval.do_normalize = False
        vis_processors, txt_processors = load_preprocess(processor_cfg)
        print(vis_processors["eval"].transform)
        print("Done!")
        
        return vis_processors, txt_processors, model

    
    def generate_response(self, prompt, image): 
        INSTRUCTION_TEMPLATE = {
            "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
            "instructblip": "<ImageHere><question>",
            "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
            "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
            "llava-1.5": "USER: <ImageHere> <question>"
        }
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        norm = transforms.Normalize(mean, std)
        raw_image = image
        
        raw_image = raw_image.convert("RGB")
        image = self.vis_processors["eval"](raw_image).unsqueeze(0)
        image = image.to("cuda")

        qu = prompt
        template = INSTRUCTION_TEMPLATE["llava-1.5"]
        qu = template.replace("<question>", qu)
        print("prompt: ", qu)
        with torch.inference_mode():
            with torch.no_grad():
                out1 = self.model.generate(
                    {"image": norm(image), "prompt":qu}, 
                    use_nucleus_sampling=False, 
                    num_beams=5,
                    max_new_tokens=512,
                    output_attentions=True,
                    opera_decoding=True,
                    scale_factor=50,
                    threshold=15.0,
                    num_attn_candidates=5,
                )
        print("OPERA's output:")
        print(out1[0])
        
        return out1
    
def get_model(model_name, model_size, model_path, device_map):
    model_handlers = {
        # "yivl": YiVLModelHandler,
        # "minicpm": MiniCPMHandler,
        # "cogvlm2": CogVLM2Handler,
        # "llavanext": LlavaNextHandler,
        "operallava": OperaLLaVAHandler,
    }
    
    if model_name.lower() in model_handlers:
        return model_handlers[model_name.lower()](model_name, model_size, model_path, device_map)
    else:
        raise ValueError(f"Model {model_name} not recognized. Available models: {list(model_handlers.keys())}")