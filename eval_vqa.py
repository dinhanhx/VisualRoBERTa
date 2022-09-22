from src.data import VisualQuestionAnswer, VisualQuestionAnswerCollator
from src.tokenization import BunTokenizer
from src.vision_language import ImageTextForCausalLM, ImageTextConfig
from pathlib import Path
from tqdm import tqdm
import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import resize
from transformers.tokenization_utils_base import BatchEncoding
import json


def process_raw_data(dp, tokenizer: BunTokenizer, image_size: list, patch_size: list):
    """ To process a data point from VQA dataloader
    for the inference phase of ImageTextCasualLM
    """
    text_inputs = tokenizer(dp['question'] + ' ? ', return_tensors='pt')
    image_inputs = torch.stack([resize(read_image(str(dp['img_file']),
                                                  ImageReadMode.RGB),
                                       image_size)],
                               0).float()
    num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])

    # Extend the shape of text_inputs.attention_masks to cover image_inputs
    extra_attention_mask = torch.ones(1, num_patches, dtype=text_inputs.attention_mask.dtype)
    attention_mask = torch.cat((text_inputs.attention_mask[:,:-1], extra_attention_mask), dim=1)  # [:,:-1] to ignore [SEP] token to enable sentence completion

    return BatchEncoding({'input_ids': text_inputs.input_ids[:,:-1],  # [:,:-1] to ignore [SEP] token to enable sentence completion
                          'attention_mask': attention_mask,
                          'token_type_ids': text_inputs.token_type_ids[:,:-1],  # [:,:-1] to ignore [SEP] token to enable sentence completion
                          'image_input': image_inputs})


def prettify_output(output, tokenizer: BunTokenizer, return_str: bool = True):
    question_mark_index = (output[0] == 4042).nonzero(as_tuple=True)[0].item()
    first_sep_index = (output[0] == 4).nonzero(as_tuple=True)[0][0].item()
    answer = output[0][question_mark_index+1: first_sep_index] if return_str else output[0]
    return tokenizer.decode(answer)


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
