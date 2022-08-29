from typing import Optional, List
import logging

import torch
from torch import nn

from transformers.models.roberta.modeling_roberta import (RobertaEmbeddings,
                                                          RobertaEncoder,
                                                          RobertaPreTrainedModel,
                                                          RobertaPooler,
                                                          RobertaLMHead)
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import BertForPreTrainingOutput

from src.image_embedding import PatchEmbedding, RegionEmbedding


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


class ImageTextForPretraining(RobertaPreTrainedModel):
    """Modified from class RobertaForMaskedLM
        to be similiar to BertForPretraining
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
