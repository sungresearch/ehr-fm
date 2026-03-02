from __future__ import annotations

from typing import Any

import transformers


class DenseTransformerConfig(transformers.PretrainedConfig):
    def __init__(
        self,
        vocab_size: int = 32768,
        hidden_size: int = 768,
        intermediate_size: int = 1152,  # 1.5 * hidden_size
        n_heads: int = 12,
        n_layers: int = 6,
        attention_width: int = 128,
        use_normed_ages: bool = False,
        use_bias: bool = False,
        hidden_act: str = "swiglu",
        remove_first_block_norm: bool = True,
        alternating_dense_layers: bool = False,
        dense_every_n_layers: int = 3,
        rope_base_sparse: float = 100.0,
        rope_base_global: float = 10000.0,
        separate_rope_by_attention: bool = False,
        **kwargs,
    ) -> None:
        """Defines a configuration for an EHRFM Transformer.

        Arguments:
            vocab_size: The number of tokens in the vocabulary
            hidden_size: The internal representation size
            intermediate_size: The size of the FFN in the transformer layers
            n_heads: The number of attention heads
            n_layers: The number of transformer encoder layers
            attention_width: EHRFM by default uses a local attention transformer with a width defined here
            use_normed_ages: Whether or not to provide normalized ages as a feature to the model
            use_bias: Whether or not to use bias terms in the transformer layers
            hidden_act: The type of activation function to use in the transformer
            remove_first_block_norm: Whether or not to remove the redundant normalization layer from the first block
            alternating_dense_layers: Whether to alternate between dense (global) and sparse (local) attention layers
            dense_every_n_layers: Frequency of dense attention layers when alternating_dense_layers is True
            rope_base_sparse: Base value for RoPE frequency in sparse attention layers. With linspace(0,2),
                            effective theta = base^2. Default 100 gives theta ~10K (standard RoPE)
            rope_base_global: Base value for RoPE frequency in global attention layers. Default 10000 gives
                            theta ~100M (extended range for long sequences)
            separate_rope_by_attention: If True, use different RoPE configurations for sparse vs global attention.
                                      If False, use rope_base_global for all layers
        """
        super().__init__(**kwargs)

        if alternating_dense_layers and dense_every_n_layers is None:
            raise ValueError("dense_every_n_layers must be specified if alternating_dense_layers is True")

        self.vocab_size = vocab_size

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.attention_width = attention_width

        self.use_normed_ages = use_normed_ages

        self.use_bias = use_bias
        self.hidden_act = hidden_act

        self.remove_first_block_norm = remove_first_block_norm
        self.alternating_dense_layers = alternating_dense_layers
        self.dense_every_n_layers = dense_every_n_layers

        self.rope_base_sparse = rope_base_sparse
        self.rope_base_global = rope_base_global
        self.separate_rope_by_attention = separate_rope_by_attention


class EHRFMConfig(transformers.PretrainedConfig):
    """A model config is defined as the combination of a transformer config and a task config."""

    def __init__(
        self,
        transformer: dict[str, Any] | None = None,
        task: dict[str, Any] | None = None,
        **kwargs,
    ):
        """A combination of a transformer config and a task config.

        It is possible to initialize this with only a transformer config, in which
        case the model will be configured for inference only.
        """
        super().__init__(**kwargs)
        if transformer is None:
            transformer = {}
        self.transformer = DenseTransformerConfig(**transformer)
        self.task = task
