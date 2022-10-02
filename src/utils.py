import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import resize
from transformers.tokenization_utils_base import BatchEncoding

from src.vision_language import ImageTextConfig
from src.tokenization import BunTokenizer


def prepare_vl_inputs(text_inputs: BatchEncoding, image_input: torch.Tensor, config: ImageTextConfig):
    """ Extend the shape of text_inputs.attention_masks
    TODO: remove me/end me
    """
    assert text_inputs.input_ids.shape[0] == image_input.shape[0]
    batch_size = image_input.shape[0]
    image_h, image_w = config.image_size
    patch_h, patch_w = config.patch_size
    num_patches = (image_h // patch_h) * (image_w // patch_w)

    extra_attention_masks = torch.ones(batch_size, num_patches, dtype=text_inputs.attention_mask.dtype)
    text_inputs.data['attention_mask'] = torch.cat((text_inputs.attention_mask, extra_attention_masks), dim=1)  # type: ignore

    return text_inputs


class ImageCaptioningTools:
    """Tools for pre-process and post-forward
    for inference mode of ImageTextCausalLM
    """

    @staticmethod
    def process_raw_data(dp, tokenizer: BunTokenizer, image_size: list, patch_size: list):
        """ To process a data point which is LIKELY from IC dataloader
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
        # [:,:-1] to ignore [SEP] token to enable sentence completion
        attention_mask = torch.cat((text_inputs.attention_mask[:, :-1], extra_attention_mask), dim=1)

        return BatchEncoding({'input_ids': text_inputs.input_ids[:, :-1],
                              'attention_mask': attention_mask,
                              'token_type_ids': text_inputs.token_type_ids[:, :-1],
                              'image_input': image_inputs})

    @staticmethod
    def prettify_output(output, tokenizer: BunTokenizer, return_str: bool = True):
        """ To remove everything that is behind [SEP]
        """
        first_sep_index = (output[0] == tokenizer.sep_token_id).nonzero(as_tuple=True)[0][0]
        caption = output[0][1: first_sep_index]
        if return_str:
            return tokenizer.decode(caption)
        else:
            return caption


class VisualQuestionAnswerTools:
    """Tools for pre-process and post-forward
    for inference mode of ImageTextCausalLM
    """

    @staticmethod
    def process_raw_data(dp, tokenizer: BunTokenizer, image_size: list, patch_size: list):
        """ To process a data point which is LIKELY from VQA dataloader
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
        attention_mask = torch.cat((text_inputs.attention_mask[:, :-1], extra_attention_mask), dim=1)
        # [:,:-1] to ignore [SEP] token to enable sentence completion

        return BatchEncoding({'input_ids': text_inputs.input_ids[:, :-1],
                              'attention_mask': attention_mask,
                              'token_type_ids': text_inputs.token_type_ids[:, :-1],
                              'image_input': image_inputs})

    @staticmethod
    def prettify_output(output, tokenizer: BunTokenizer, return_str: bool = True):
        """ To remove everything that is behind [SEP]
        and that is before question mark

        WARNING: question_mark is hardcoded
        """
        question_mark_index = (output[0] == 4042).nonzero(as_tuple=True)[0][0]
        first_sep_index = (output[0] == tokenizer.sep_token_id).nonzero(as_tuple=True)[0][0]
        answer = output[0][question_mark_index+1: first_sep_index]
        if return_str:
            return tokenizer.decode(answer)
        else:
            return answer


class ImageTextMatchingTools:
    """ Tools for pre-process and post-forward
    for inference mode of ImageTextForPretraining
    """

    @staticmethod
    def process_raw_data(dp, tokenizer: BunTokenizer, image_size: list, patch_size: list):
        text_inputs = tokenizer(dp['caption'], return_tensors='pt')
        image_inputs = torch.stack([resize(read_image(str(dp['img_file']),
                                                      ImageReadMode.RGB),
                                           image_size)],
                                   0).float()
        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        # Extend the shape of text_inputs.attention_masks to cover image_inputs
        extra_attention_mask = torch.ones(1, num_patches, dtype=text_inputs.attention_mask.dtype)
        # [:,:-1] to ignore [SEP] token to enable sentence completion
        attention_mask = torch.cat((text_inputs.attention_mask[:, :-1], extra_attention_mask), dim=1)

        return BatchEncoding({'input_ids': text_inputs.input_ids[:, :-1],
                              'attention_mask': attention_mask,
                              'token_type_ids': text_inputs.token_type_ids[:, :-1],
                              'image_input': image_inputs})

    @staticmethod
    def prettify_output(output):
        seq_relationship_logits = output['seq_relationship_logits']
        if seq_relationship_logits[0, 0] < seq_relationship_logits[0, 1]:  # this is reversed of BERT NSP
            return True
        else:
            return False
