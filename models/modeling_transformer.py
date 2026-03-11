"""
Hugging Face PreTrainedModel wrapper for the custom Transformer model.
Wrap the original Transformer class so that lm-evaluation-harness can load the model via AutoModelForCausalLM.
"""

import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_transformer import TransformerHuggingFaceConfig
from .transformer import Transformer, ModelConfig


class TransformerHuggingFace(PreTrainedModel):
    config_class = TransformerHuggingFaceConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False

    def __init__(self, config: TransformerHuggingFaceConfig):
        super().__init__(config)

        model_config = ModelConfig(
            vocab_size=config.vocab_size,
            seq_len=config.seq_len,
            dim=config.dim,
            expand=config.expand,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            mlp=config.mlp,
            rmsnorm_eps=config.rmsnorm_eps,
            tie_embeddings=config.tie_embeddings,
        )

        self.model = Transformer(model_config)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.model.lm_head = new_embeddings

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        logits = self.model(input_ids, attn_mask=None)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}
