from src.data import ImageTextPair
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
    """ To process a data point from IC dataloader
    for the inference phase of ImageTextCasualLM
    """
    text_inputs = tokenizer(' ', return_tensors='pt')
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
    first_sep_index = (output[0] == 4).nonzero(as_tuple=True)[0][0]
    caption = output[0][1: first_sep_index]
    if return_str:
        return tokenizer.decode(caption)
    else:
        return caption


ic = ImageTextPair(Path('/home/dinhanhx/data/'),
                   Path('/home/dinhanhx/data/TranslateCOCO2017/'),
                   split='test_uit_viic',
                   do_sort=True)
bun_tokenizer = BunTokenizer.from_pretrained('vinai/bartpho-syllable')

ver = 1
config = ImageTextConfig.from_json_file('assets/imagetext-casual-base-config.json')
model = ImageTextForCausalLM(config)
model.load_state_dict(torch.load(f'uit_viic_logs/lightning_logs/version_{ver}/checkpoints/uit_viic.pt'))
model = model.eval()

lic = len(ic)
d = {}
cur_image_id = -1
counter = 0
for i in tqdm(range(lic)):
    dp = ic[i]
    if cur_image_id == dp['image_id']:
        continue
    else:
        cur_image_id = dp['image_id']
        inputs = process_raw_data(dp, bun_tokenizer, config.image_size, config.patch_size)
        with torch.no_grad():
            output = model.generate(inputs=inputs['input_ids'],
                                    max_new_tokens=40,
                                    num_beams=5,
                                    do_sample=False,
                                    early_stopping=True,
                                    attention_mask=inputs['attention_mask'],
                                    token_type_ids=inputs['token_type_ids'],
                                    image_input=inputs['image_input'])
            predict = prettify_output(output, bun_tokenizer)
            d[f'{counter}'] = {'predict': predict,
                            'image_id': dp['image_id']}
            counter += 1

with open('test_uit_viic.json', 'w', encoding='utf-8') as tf:
    json.dump(d, tf, ensure_ascii=False)
