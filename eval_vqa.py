from src.data import VisualQuestionAnswer, VisualQuestionAnswerCollator
from src.tokenization import BunTokenizer
from src.vision_language import ImageTextForCausalLM, ImageTextConfig
from src.utils import VisualQuestionAnswerTools
from pathlib import Path
from tqdm import tqdm
import torch
import json
from typing import Callable


process_raw_data: Callable = VisualQuestionAnswerTools.process_raw_data
prettify_output: Callable = VisualQuestionAnswerTools.prettify_output


vqa = VisualQuestionAnswer(Path('/home/dinhanhx/data/'),
                           Path('/home/dinhanhx/data/ViVQA-main'),
                           split='test')
bun_tokenizer = BunTokenizer.from_pretrained('vinai/bartpho-syllable')

ver = 1
config = ImageTextConfig.from_json_file('assets/imagetext-casual-base-config.json')
model = ImageTextForCausalLM(config)
model.load_state_dict(torch.load(f'vqa_logs/lightning_logs/version_{ver}/checkpoints/vqa.pt'))
model = model.eval()

ldp = len(vqa)
d = {}
for i in tqdm(range(ldp)):
    dp = vqa[i]
    inputs = process_raw_data(dp, bun_tokenizer, config.image_size, config.patch_size)
    with torch.no_grad():
        output = model.generate(inputs=inputs['input_ids'],
                                max_new_tokens=10,
                                num_beams=5,
                                do_sample=False,
                                early_stopping=True,
                                attention_mask=inputs['attention_mask'],
                                token_type_ids=inputs['token_type_ids'],
                                image_input=inputs['image_input'])
    predict = prettify_output(output, bun_tokenizer)
    reference: str = dp['answer']
    correct = None
    if predict.lower() == reference.lower():
        correct = True
    else:
        correct = False

    d[f'{i}'] = {'correct': correct,
                 'predict': predict,
                 'reference': reference}

with open('test_vqa.json', 'w', encoding='utf-8') as tf:
    json.dump(d, tf, ensure_ascii=False)
