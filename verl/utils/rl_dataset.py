# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

import requests
import torch
from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.models.transformers.qwen2_5_vl import get_rope_index


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        if key not in ["pixel_values", "image_grid_thw"]:
            tensors[key] = torch.stack(value, dim=0)

    return {**tensors, **non_tensors}


def process_image(image: ImageObject, max_pixels: int, min_pixels: int) -> ImageObject:
    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def translate(text: str, language_code: str) -> str:
    API_URL = "http://localhost:1314/generate"
    payload = {"text": text, "language_code": language_code}
    response = requests.post(API_URL, json=payload)
    return response.json()["response"]


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key="prompt",
        max_prompt_length=1024,
        truncation="error",
        max_pixels=None,
        min_pixels=None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        self.dataset = load_dataset(data_path, split=data_split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataset[index]
        messages = [
            {"role": "system", "content": r"Please reason step by step, and put your final answer within \boxed{}."},
            {"role": "user", "content": row_dict[self.prompt_key]},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        if "images" in row_dict:  # expand image token
            raw_prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            row_dict["images"] = [
                process_image(image, self.max_pixels, self.min_pixels) for image in row_dict["images"]
            ]
            image_inputs = self.processor.image_processor(row_dict["images"], return_tensors="pt")
            image_grid_thw = image_inputs["image_grid_thw"]
            row_dict.update(image_inputs)

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while "<image>" in prompt:
                    prompt = prompt.replace(
                        "<image>",
                        "<|vision_start|>"
                        + "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length)
                        + "<|vision_end|>",
                        1,
                    )
                    index += 1

                prompt = prompt.replace("<|placeholder|>", self.processor.image_token)
        else:
            raw_prompt = prompt

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if "images" in row_dict:
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )  # (3, seq_len)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seqlen,)

        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        return row_dict


class RLHFVQADataset(Dataset):
    """
    For multipe-choice VQA datasets.
    Example: https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key="question",
        options_key="choices",
        max_prompt_length=1024,
        truncation="error",
        max_pixels=None,
        min_pixels=None,
        is_validation=False,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.options_key = options_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.is_validation = is_validation

        # HACK
        self.target_languages = ["zh", "he"]
        self.system_prompts = {
            "zh": r"请逐步分析并解释你的思考过程，最后将最终答案（A、B、C或D）标注在\boxed{}中。",
            "he": r"אנא פרט את השיקולים שלך והסבר את הפתרון שלך בצעדים, והכנס את התשובה הסופית שלך בתוך \boxed{} (A, B, C או D).",
        }

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        self.dataset = load_dataset(data_path, split=data_split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataset[index]

        question = row_dict[self.prompt_key]
        if "<image>" not in question:
            question = f"<image>{question}"
        self.options_key = "choices" if "choices" in row_dict else "options"
        options = row_dict[self.options_key]
        question += " " + " ".join([f"({chr(i + ord('A'))}) {option}" for i, option in enumerate(options)])

        messages = [
            {"role": "system", "content": r"Please reason step by step, and put your final answer within \boxed{} (A, B, C, or D)."},
            {"role": "user", "content": question},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        image_key = "image" if "image" in row_dict else "images"

        raw_prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        if not isinstance(row_dict[image_key], list):
            row_dict[image_key] = [row_dict[image_key]]
        row_dict[image_key] = [
            process_image(image, self.max_pixels, self.min_pixels) for image in row_dict[image_key]
        ]
        image_inputs = self.processor.image_processor(row_dict[image_key], return_tensors="pt")
        image_grid_thw = image_inputs["image_grid_thw"]
        row_dict.update(image_inputs)

        if image_grid_thw is not None:
            merge_length = self.processor.image_processor.merge_size**2
            index = 0
            while "<image>" in prompt:
                prompt = prompt.replace(
                    "<image>",
                    "<|vision_start|>"
                    + "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length)
                    + "<|vision_end|>",
                    1,
                )
                index += 1

            prompt = prompt.replace("<|placeholder|>", self.processor.image_token)

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        position_ids = get_rope_index(
            self.processor,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )  # (3, seq_len)

        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        if self.is_validation:
            return row_dict

        # translate the English question to other languages
        language = random.choice(self.target_languages)
        question_translated = translate(question.replace("<image>", ""), language)
        question_translated = f"<image>{question_translated}"
        messages_translated = [
            {"role": "system", "content": self.system_prompts[language]},
            {"role": "user", "content": question_translated},
        ]
        prompt_translated = self.tokenizer.apply_chat_template(messages_translated, add_generation_prompt=True, tokenize=False)
        raw_prompt_translated = prompt_translated.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        if image_grid_thw is not None:
            merge_length = self.processor.image_processor.merge_size**2
            index = 0
            while "<image>" in prompt_translated:
                prompt_translated = prompt_translated.replace(
                    "<image>",
                    "<|vision_start|>"
                    + "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length)
                    + "<|vision_end|>",
                    1,
                )
                index += 1

            prompt_translated = prompt_translated.replace("<|placeholder|>", self.processor.image_token)

        input_ids_translated, attention_mask_translated = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_translated,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        position_ids_translated = get_rope_index(
            self.processor,
            input_ids=input_ids_translated,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask_translated,
        )

        row_dict["input_ids_translated"] = input_ids_translated
        row_dict["attention_mask_translated"] = attention_mask_translated
        row_dict["position_ids_translated"] = position_ids_translated
        row_dict["raw_prompt_ids_translated"] = self.tokenizer.encode(raw_prompt_translated, add_special_tokens=False)
        row_dict["language"] = language

        return row_dict
