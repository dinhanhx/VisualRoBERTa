from typing import Optional, List, Tuple, Union
import logging
import warnings

import torch
from torch import nn
import torch.distributed as dist

from transformers.models.roberta.modeling_roberta import (RobertaEmbeddings,
                                                          RobertaEncoder,
                                                          RobertaPreTrainedModel,
                                                          RobertaPooler,
                                                          RobertaLMHead)
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.modeling_outputs import (BaseModelOutputWithPoolingAndCrossAttentions,
                                           CausalLMOutputWithCrossAttentions)
from transformers.models.bert.modeling_bert import BertForPreTrainingOutput

from src.image_embedding import PatchEmbedding, RegionEmbedding

# >> BO Import for Generation << #
from transformers.generation_logits_process import LogitsProcessorList
from transformers.generation_stopping_criteria import (StoppingCriteriaList,
                                                       validate_stopping_criteria)
from transformers.generation_utils import (GreedySearchOutput,
                                           GreedySearchDecoderOnlyOutput,
                                           GreedySearchEncoderDecoderOutput,
                                           BeamSearchOutput)
from transformers.generation_beam_search import (BeamScorer)
# >> EO Import for Generation << #


class ImageTextConfig(RobertaConfig):
    model_type: str = 'imagetext'
    def __init__(self, image_size=[480, 512], image_embedding_type='patch', patch_size=[32, 32],
                 pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs):
        super().__init__(pad_token_id, bos_token_id, eos_token_id, **kwargs)
        self.image_size = image_size
        self.image_embedding_type = image_embedding_type
        self.patch_size = patch_size


class ImageTextModel(RobertaPreTrainedModel):
    def __init__(self, config: ImageTextConfig, add_pooling_layer=True):
        super().__init__(config)

        self.text_embedding = RobertaEmbeddings(config)
        if config.image_embedding_type == 'patch':
            self.image_embedding = PatchEmbedding(image_size=config.image_size,
                                                  patch_size=config.patch_size,
                                                  emb_dim=config.hidden_size)
        elif config.image_embedding_type == 'region':
            self.image_embedding = RegionEmbedding()

        self.econder = RobertaEncoder(config)
        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        self.post_init()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                image_input: Optional[torch.Tensor] = None):

        """
        Copied from class RoBERTaModel
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.text_embedding, "token_type_ids"):
                buffered_token_type_ids = self.text_embedding.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.text_embedding(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        """
        End of Copied from class RoBERTaModel
        """

        image_embedding_output = self.image_embedding(image_input)

        concat_output = torch.cat((embedding_output, image_embedding_output), dim=1)

        # To cover image_input
        # To fix self.get_extended_attention_mask(attention_mask, input_shape)
        # when self.config.is_decoder
        # it's somewhat magic, don't touch it
        if self.config.is_decoder:
            extended_attention_mask = __class__.correct_extend_attention_mask(input_shape=input_shape,
                                                                              attention_mask_=attention_mask,
                                                                              num_patches_=self.get_num_patches,
                                                                              model_dtype=self.dtype)

        encoder_outputs = self.econder(
            concat_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions
        )

    @property
    def get_num_patches(self):
        return (self.config.image_size[0] // self.config.patch_size[0]) \
            * (self.config.image_size[1] // self.config.patch_size[1])

    @staticmethod
    def correct_extend_attention_mask(input_shape: torch.Size,
                                      attention_mask_: torch.Tensor,
                                      num_patches_: int,
                                      model_dtype,
                                      optimize: bool = True):
        device = attention_mask_.device
        dtype = attention_mask_.dtype
        batch_size, seq_length = input_shape

        # Create triangle mask for text modality
        triangle_mask = torch.tril(
            torch.ones((seq_length, seq_length),
                       dtype=dtype,
                       device=device)
        )[None, :, :].repeat((batch_size, 1, 1))

        extended_attention_mask: torch.Tensor = torch.Tensor()
        if triangle_mask.shape[1] == attention_mask_[:, :-num_patches_].shape[1]:
            # Filtering ...
            filtered_triangle_mask = triangle_mask * attention_mask_[:, :seq_length][:, None, :]

            # Create full attention mask for text modality to image modality
            text_to_image_mask = torch.ones((batch_size, seq_length, num_patches_),
                                            dtype=dtype,
                                            device=device)
            triangle_text_to_image_mask = torch.concat((filtered_triangle_mask, text_to_image_mask), dim=-1)

            # Create extended_attention_mask with zeros as placeholders
            extended_attention_mask = torch.zeros((batch_size, seq_length+num_patches_, seq_length+num_patches_),
                                                  dtype=dtype,
                                                  device=device)
            extended_attention_mask[:, :seq_length, :].copy_(triangle_text_to_image_mask)
            # Filtering ...
            extended_attention_mask = (extended_attention_mask.transpose(-1, -2) * attention_mask_[:, None, :]).transpose(-1, -2)  # noqa

            # Add full attention mask for image modality
            extended_attention_mask[:, seq_length:, seq_length:].copy_(
                torch.ones((batch_size, num_patches_, num_patches_),
                           dtype=dtype,
                           device=device)
            )
        else:
            # handle when `past_key_values` are used
            # prefix ones mask added in front of triangle_mask
            prefix_seq_len = attention_mask_[:, :-num_patches_].shape[1] - triangle_mask.shape[1]
            triangle_mask = torch.cat(
                [
                    torch.ones((batch_size, seq_length, prefix_seq_len),
                               dtype=dtype,
                               device=device),
                    triangle_mask
                ],
                dim=-1
            )
            # Create full attention mask for text modality to image modality
            text_to_image_mask = torch.ones((batch_size, seq_length, num_patches_),
                                            dtype=dtype,
                                            device=device)
            triangle_text_to_image_mask = torch.concat((triangle_mask, text_to_image_mask), dim=-1)

            # Create extended_attention_mask with zeros as placeholders
            extended_attention_mask = torch.zeros((batch_size, seq_length+num_patches_,
                                                   seq_length+prefix_seq_len+num_patches_),
                                                  dtype=dtype,
                                                  device=device)
            extended_attention_mask[:, :seq_length, :].copy_(triangle_text_to_image_mask)
            extended_attention_mask[:, seq_length:, seq_length+prefix_seq_len:].copy_(
                torch.ones((batch_size, num_patches_, num_patches_),
                           dtype=dtype,
                           device=device)
            )

        extended_attention_mask = extended_attention_mask[:, None, :, :]
        if optimize:
            extended_attention_mask = extended_attention_mask.to(dtype=model_dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(model_dtype).min

        return extended_attention_mask


class ImageTextForPretraining(RobertaPreTrainedModel):
    """ Modified from class RobertaForMaskedLM
        to be similar to BertForPretraining
    """
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config: ImageTextConfig):
        super().__init__(config)

        if config.is_decoder:
            logging.warning("If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False")

        self.imagetext = ImageTextModel(config)
        self.lm_head = RobertaLMHead(config)
        self.imagetext_relations = nn.Linear(config.hidden_size, 2)

        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])
        self.post_init()

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.FloatTensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                match_labels: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                image_input: Optional[torch.Tensor] = None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.imagetext(input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 position_ids=position_ids,
                                 head_mask=head_mask,
                                 inputs_embeds=inputs_embeds,
                                 encoder_hidden_states=encoder_hidden_states,
                                 encoder_attention_mask=encoder_attention_mask,
                                 output_attentions=output_attentions,
                                 output_hidden_states=output_hidden_states,
                                 return_dict=return_dict,
                                 image_input=image_input)

        sequence_output, pooled_output = outputs[:2]
        prediction_scores = self.lm_head(sequence_output)
        imagetext_relations_score = self.imagetext_relations(pooled_output)

        total_loss = None
        if labels is not None and match_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(imagetext_relations_score.view(-1, 2), match_labels.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, imagetext_relations_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=imagetext_relations_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ImageTextForCausalLM(RobertaPreTrainedModel):
    """ Copied from RobertaForCausalLM
    """
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config: ImageTextConfig):
        super().__init__(config)

        if not config.is_decoder:
            logging.warning('If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True.`')

        self.imagetext = ImageTextModel(config)
        self.lm_head = RobertaLMHead(config)

        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])
        self.post_init()

        # To store raw image
        # See self.prepare_inputs_for_generation()
        self.__image_input_cache__ = None

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.FloatTensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,  # type: ignore
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                image_input: Optional[torch.Tensor] = None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.imagetext(input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 position_ids=position_ids,
                                 head_mask=head_mask,
                                 inputs_embeds=inputs_embeds,
                                 encoder_hidden_states=encoder_hidden_states,
                                 encoder_attention_mask=encoder_attention_mask,
                                 past_key_values=past_key_values,
                                 use_cache=use_cache,
                                 output_attentions=output_attentions,
                                 output_hidden_states=output_hidden_states,
                                 return_dict=return_dict,
                                 image_input=image_input)

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()  # type: ignore
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size),
                               labels.view(-1))  # type: ignore

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, past=None, **model_kwargs):
        if past is not None:
            # To handle the inputs with past
            assert type(self.__image_input_cache__) is not None, ("self.__image_input_cache is None. "
                                                                  "Check `image_input` in your `inputs`, "
                                                                  "Or something wrong with this function"
                                                                  "when being called for first time.")

            batch_length = input_ids.shape[0]
            num_patches = self.get_num_patches

            last_input_ids = input_ids[:, -1:]
            attention_mask = torch.ones_like(input_ids)
            last_token_type_ids = torch.zeros_like(last_input_ids)

            # Extend the shape of last_attention_masks to cover image_inputs
            extra_attention_mask = torch.ones(batch_length, num_patches, dtype=attention_mask.dtype)
            attention_mask = torch.cat((attention_mask, extra_attention_mask), dim=1)

            # Remove image modality from `past_key_values`
            # Because `past` is (text, image) and when Transformer (in this scope, RobertaEncoder)
            # appends new token, new token is appended to `past`
            # `past` becomes (text, image, new token) hence it breaks the sequence of text
            # Therefore, `image` needs to be removed.
            # HOWEVER this action makes the model compute more slowly because it has to recompute image modality
            # for people whoever work on this code, please try to make things faster
            # probably need to change the order of inputs in torch.cat in `forward()` in `ImageTextModel`
            # then probably need to retrain the model
            new_past = tuple((tuple((past_key[:, :, :-num_patches, :] for past_key in layer)) for layer in past))
            return {'input_ids': last_input_ids,
                    'past_key_values': new_past,
                    'attention_mask': attention_mask,
                    'token_type_ids': last_token_type_ids,
                    'image_input': self.__image_input_cache__}
        else:
            # To handle the very first inputs

            # Handle beam search case
            self.__image_input_cache__ = model_kwargs.get('image_input', None).repeat(input_ids.shape[0], 1, 1, 1)
            return {'input_ids': input_ids,
                    'attention_mask': model_kwargs.get('attention_mask', None),
                    'token_type_ids': model_kwargs.get('token_type_ids', None),
                    'image_input': self.__image_input_cache__}

    def _reorder_cache(self, past, beam_idx):
        """ Copied from RoBERTaForCasualLM
            You may find the implementation of GPT-2's one a bit different
            however both do same things
            https://github.com/huggingface/transformers/blob/983e40ac3b2af68fd6c927dce09324d54d023e54/src/transformers/models/gpt2/modeling_gpt2.py#L1104-L1114  # noqa
        """
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    @property
    def get_num_patches(self):
        return (self.config.image_size[0] // self.config.patch_size[0]) \
            * (self.config.image_size[1] // self.config.patch_size[1])

    def greedy_search(self,
                      input_ids: torch.LongTensor,
                      logits_processor: Optional[LogitsProcessorList] = None,
                      stopping_criteria: Optional[StoppingCriteriaList] = None,
                      max_length: Optional[int] = None,
                      pad_token_id: Optional[int] = None,
                      eos_token_id: Optional[int] = None,
                      output_attentions: Optional[bool] = None,
                      output_hidden_states: Optional[bool] = None,
                      output_scores: Optional[bool] = None,
                      return_dict_in_generate: Optional[bool] = None,
                      synced_gpus: Optional[bool] = False, **model_kwargs) \
                        -> Union[GreedySearchOutput, torch.LongTensor]:
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]

        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -(1+self.get_num_patches), :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids

    def beam_search(self,
                    input_ids: torch.LongTensor,
                    beam_scorer: BeamScorer,
                    logits_processor: Optional[LogitsProcessorList] = None,
                    stopping_criteria: Optional[StoppingCriteriaList] = None,
                    max_length: Optional[int] = None,
                    pad_token_id: Optional[int] = None,
                    eos_token_id: Optional[int] = None,
                    output_attentions: Optional[bool] = None,
                    output_hidden_states: Optional[bool] = None,
                    output_scores: Optional[bool] = None,
                    return_dict_in_generate: Optional[bool] = None,
                    synced_gpus: Optional[bool] = False, **model_kwargs) \
                        -> Union[BeamSearchOutput, torch.LongTensor]:
        """ TODO - implement
        """
        raise NotImplementedError
