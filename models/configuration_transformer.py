"""Hugging Face PretrainedConfig for the custom Transformer model."""

from transformers import PretrainedConfig


class TransformerHuggingFaceConfig(PretrainedConfig):
    model_type = "transformer_hugging_face"

    def __init__(
        self,
        vocab_size=50277,
        seq_len=2048,
        dim=320,
        expand=2.6666666666666665,
        n_layers=7,
        n_heads=10,
        mlp="glu",
        rmsnorm_eps=1e-6,
        tie_embeddings=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dim = dim
        self.expand = expand
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.mlp = mlp
        self.rmsnorm_eps = rmsnorm_eps
        self.tie_embeddings = tie_embeddings
        super().__init__(tie_word_embeddings=tie_embeddings, **kwargs)
