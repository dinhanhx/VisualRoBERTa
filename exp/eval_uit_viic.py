import json
from pathlib import Path
from typing import Callable

import torch
from tqdm import tqdm

from src.data import ImageTextPair
from src.tokenization import BunTokenizer
from src.utils import ImageCaptioningTools
from src.vision_language import ImageTextConfig, ImageTextForCausalLM

process_raw_data: Callable = ImageCaptioningTools.process_raw_data
prettify_output: Callable = ImageCaptioningTools.prettify_output


ic = ImageTextPair(
    Path("/home/dinhanhx/data/"),
    Path("/home/dinhanhx/data/TranslateCOCO2017/"),
    split="test_uit_viic",
    do_sort=True,
)
bun_tokenizer = BunTokenizer.from_pretrained("vinai/bartpho-syllable")

ver = 1
config = ImageTextConfig.from_json_file("assets/imagetext-casual-base-config.json")
model = ImageTextForCausalLM(config)  # type: ignore
model.load_state_dict(
    torch.load(f"uit_viic_logs/lightning_logs/version_{ver}/checkpoints/uit_viic.pt")
)
model = model.eval()

lic = len(ic)
d = {}
cur_image_id = -1
counter = 0
for i in tqdm(range(lic)):
    dp = ic[i]
    if cur_image_id == dp["image_id"]:
        continue
    else:
        cur_image_id = dp["image_id"]
        inputs = process_raw_data(
            dp, bun_tokenizer, config.image_size, config.patch_size
        )
        with torch.no_grad():
            output = model.generate(
                inputs=inputs["input_ids"],  # type: ignore
                max_new_tokens=40,
                num_beams=5,
                do_sample=False,
                early_stopping=True,
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs["token_type_ids"],
                image_input=inputs["image_input"],
            )
            predict = prettify_output(output, bun_tokenizer)
            d[f"{counter}"] = {"predict": predict, "image_id": dp["image_id"]}
            counter += 1

with open("assets/test_uit_viic.json", "w", encoding="utf-8") as tf:
    json.dump(d, tf, ensure_ascii=False)
