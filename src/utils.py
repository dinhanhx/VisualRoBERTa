import torch
from transformers.tokenization_utils_base import BatchEncoding

from src.vision_language import ImageTextConfig


def prepare_vl_inputs(text_inputs: BatchEncoding, image_input: torch.Tensor, config: ImageTextConfig):
    """ Extend the shape of text_inputs.attention_masks
    """
    assert text_inputs.input_ids.shape[0] == image_input.shape[0]
    batch_size = image_input.shape[0]
    image_h, image_w = config.image_size
    patch_h, patch_w = config.patch_size
    num_patches = (image_h // patch_h) * (image_w // patch_w)

    extra_attention_masks = torch.ones(batch_size, num_patches, dtype=text_inputs.attention_mask.dtype)
    text_inputs.data['attention_mask'] = torch.cat((text_inputs.attention_mask, extra_attention_masks), dim=1)  # type: ignore

    return text_inputs
