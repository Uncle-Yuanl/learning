#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   cln.py
@Time   :   2023/06/06 14:01:32
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   Conditional Layer Normalization
'''

import logging
import os
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
from typing import Callable, Dict, List, Optional, Set, Tuple, Union


import torch
from torch import Tensor, Size
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import AutoConfig, AutoTokenizer
from transformers.models.distilbert.modeling_distilbert import create_sinusoidal_embeddings
from transformers.models.distilbert.modeling_distilbert import DistilBertPreTrainedModel
from transformers.models.distilbert.modeling_distilbert import Transformer
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput


def align(tensor: Tensor, axes, ndim=None):
    """对齐tensor，主要用来补维度
    axes：原来的第i维对齐新tensor的第axes[i]维；
    ndim：新tensor的维度。
    """
    assert len(axes) == tensor.dim()
    assert ndim or min(axes) >= 0
    ndim = ndim or max(axes) + 1
    indices = [None] * ndim
    for i in axes:
        indices[i] = slice(None)
    return tensor[indices]


class ConditionalLayerNorm(nn.LayerNorm):
    """conditional layer norm
    cond -> affine transforme -> concat(LN, cond_result)
    """
    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
        **kwargs
    ) -> None:
        super(ConditionalLayerNorm, self).__init__(
            normalized_shape, eps,
            elementwise_affine, device,dtype
        )

        self.cond_dims = kwargs.get('cond_dims', 128)
        self.hidden_units = kwargs.get('hidden_units', 768)
        self.hidden_activation = kwargs.get('hidden_activation')

        if not self.hidden_activation or self.hidden_activation == 'linear':
            gain = 1
            self.hidden_activation = None
        else:
            gain = nn.init.calculate_gain(self.hidden_activation)
            self.hidden_activation = getattr(nn, self.hidden_activation)()
            
        self.hidden_initializer = kwargs.get('initializer', 'xavier_uniform_')
        self.hidden_initializer = getattr(nn.init, self.hidden_initializer)

        # cond dense
        self.hidden_dense = nn.Linear(
            in_features=self.cond_dims,
            out_features=self.hidden_units,
            bias=False,
            device=device
        )
        # init
        self.hidden_initializer(self.hidden_dense.weight, gain=gain)

        # gamma(*) and beta(+)
        self.register_parameter(
            name='clngma',
            param=nn.Parameter(
                data=torch.ones(self.hidden_units,),
                requires_grad=True
            )
        )

        self.register_parameter(
            name='clnbta',
            param=nn.Parameter(
                data=torch.zeros(self.hidden_units,),
                requires_grad=True
            )
        )

        # initializer = 'zeros'
        # notice: from_pretrain has func _fix_key
        self.clngma_dense = nn.Linear(
            in_features=self.hidden_units,
            out_features=self.hidden_units,
            bias=False,
            device=device
        )
        torch.nn.init.zeros_(self.clngma_dense.weight)

        self.clnbta_dense = nn.Linear(
            in_features=self.hidden_units,
            out_features=self.hidden_units,
            bias=False,
            device=device
        )
        torch.nn.init.zeros_(self.clnbta_dense.weight)

    def forward(self, inputs, conds):
        conds = self.hidden_dense(conds)
        if self.hidden_activation:
            conds = self.hidden_activation(conds)
                
        conds = align(conds, [0, -1], inputs.dim())

        gamma = self.get_parameter('clngma') + self.clngma_dense(conds)
        beta  = self.get_parameter('clnbta') + self.clnbta_dense(conds)

        inputs = inputs * gamma + beta

        return inputs


class ConditionalEmbeddings(nn.Module):
    def __init__(self, config: PretrainedConfig, **kwargs):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)
        if config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=config.max_position_embeddings, dim=config.dim, out=self.position_embeddings.weight
            )
        # conditional layer
        if "cond_size" in config.__dict__.keys():
            cond_size = config.cond_size
        else:
            cond_size = kwargs.get("cond_size", 6)
        if "cond_dims" in config.__dict__.keys():
            cond_dims = config.cond_dims
        else:
            cond_dims = kwargs.get("cond_dims", 128)

        self.cond_embeddings = nn.Embedding(
            num_embeddings=cond_size,
            embedding_dim=cond_dims
        )

        self.LayerNorm = ConditionalLayerNorm(
            config.dim, eps=1e-12,
            **kwargs
        )

        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
            self,
            input_ids: torch.Tensor, 
            condition_ids: torch.Tensor,
            input_embeds: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
        """
        Parameters:
            input_ids (torch.Tensor):
                torch.tensor(bs, max_seq_length) The token ids to embed.
            input_embeds (*optional*, torch.Tensor):
                The pre-computed word embeddings. Can only be passed if the input ids are `None`.


        Returns: torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings, no token_type
        embeddings)
        """
        if input_ids is not None:
            input_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)

        seq_length = input_embeds.size(1)

        # Setting the position-ids to the registered buffer in constructor, it helps
        # when tracing the model without passing position-ids, solves
        # isues similar to issue #5664
        if hasattr(self, "position_ids"):
            position_ids = self.position_ids[:, :seq_length]
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)

        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

        embeddings = input_embeds + position_embeddings  # (bs, max_seq_length, dim)
        
        # cond embeddings
        cond_embed = self.cond_embeddings(condition_ids)
        embeddings = self.LayerNorm(embeddings, cond_embed)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings


class ConditionalDistilBertModel(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig, **kwargs):
        super().__init__(config)

        self.embeddings = ConditionalEmbeddings(config, **kwargs)  # Embeddings
        self.transformer = Transformer(config)  # Encoder

        # Initialize weights and apply final processing
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.embeddings.position_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        num_position_embeds_diff = new_num_position_embeddings - self.config.max_position_embeddings

        # no resizing needs to be done if the length stays the same
        if num_position_embeds_diff == 0:
            return

        logger.info(f"Setting `config.max_position_embeddings={new_num_position_embeddings}`...")
        self.config.max_position_embeddings = new_num_position_embeddings

        old_position_embeddings_weight = self.embeddings.position_embeddings.weight.clone()

        self.embeddings.position_embeddings = nn.Embedding(self.config.max_position_embeddings, self.config.dim)

        if self.config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=self.config.max_position_embeddings, dim=self.config.dim, out=self.position_embeddings.weight
            )
        else:
            with torch.no_grad():
                if num_position_embeds_diff > 0:
                    self.embeddings.position_embeddings.weight[:-num_position_embeds_diff] = nn.Parameter(
                        old_position_embeddings_weight
                    )
                else:
                    self.embeddings.position_embeddings.weight = nn.Parameter(
                        old_position_embeddings_weight[:num_position_embeds_diff]
                    )
        # move position_embeddings to correct device
        self.embeddings.position_embeddings.to(self.device)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[List[int]]]):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.transformer.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        condition_ids:Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embeddings = self.embeddings(input_ids, condition_ids, inputs_embeds)  # (bs, seq_length, dim)

        return self.transformer(
            x=embeddings,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class DistilBertForSequenceClassification(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.distilbert = ConditionalDistilBertModel(config, **kwargs)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # Initialize weights and apply final processing
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        condition_ids:Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            condition_ids=condition_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )

    # def from_pretrained(self, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
    #     model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
    #     # tie self defined parameters
    #     self_defined_params_keys = [
    #         "distilbert.embeddings.LayerNorm.gamma",
    #         "distilbert.embeddings.LayerNorm.beta",
    #         "distilbert.embeddings.LayerNorm.gamma_dense.weight",
    #         "distilbert.embeddings.LayerNorm.beta_dense.weight"
    #     ]
    #     state_dict = torch.load(pretrained_model_name_or_path + '/pytorch_model.bin')
    #     for sdpk in self_defined_params_keys:
    #         if sdpk in state_dict:
    #             model.state_dict()[sdpk] = state_dict[sdpk]
    #         else:
    #             raise ValueError(f"cannot find {sdpk} in state_dict")

    #     return model
    

if __name__ == "__main__":
    pretrained = '/media/data/pretrained_models/Distilbert'
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    config = AutoConfig.from_pretrained(
        pretrained,
        num_labels=3,
        ignore_mismatched_sizes=True)
    
    kwargs = {
        "cond_size": 6,
        "cond_dims": 128
    }
    model = DistilBertForSequenceClassification(config, **kwargs)

    # test = tokenizer(["it's so good", "it's too expensive"])
    # test.update({'condition_ids': [[1], [5]]})
    # test = {k: torch.LongTensor(v) for k, v in test.items()}
    # result = model(**test)

    model = model.from_pretrained("/media/data/pretrained_models/concept/nutrition/total_ensemble/Acceptable_Costs/Ensemble_0/best")
    print()
