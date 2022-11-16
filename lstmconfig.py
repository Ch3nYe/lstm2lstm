import copy
from typing import Optional

from transformers import PretrainedConfig, BertTokenizer
from transformers import AutoConfig


class LSTMConfig(PretrainedConfig):
    model_type = "LSTM"

    def __init__(
            self,
            vocab_size: int,
            tokenizer: BertTokenizer,
            input_size: int,
            hidden_size: int = 128,
            num_layers: int = 2,
            bias: bool = True,
            batch_first: bool = False,
            dropout: float = 0.0,
            bidirectional: bool = False,
            is_decoder: bool = False,
            attn_type: str = None,
            context_gate: str = None,
            bidirectional_encoder: bool = False,
            attentional_decoder: bool = False,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.is_decoder = is_decoder
        self.attn_type = attn_type
        self.context_gate = context_gate
        self.bidirectional_encoder = bidirectional_encoder
        self.attentional_decoder = attentional_decoder
        self.pad_token_id = tokenizer.pad_token_id
        self.unk_token_id = tokenizer.unk_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id

        if is_decoder:
            self.decoder_start_token_id = self.bos_token_id


class LSTM2LSTMConfig(PretrainedConfig):
    r"""
    [`LSTM2LSTMConfig`] is the configuration class to store the configuration of a [`LSTM2LSTMModel`]. It is
    used to instantiate an Encoder Decoder model according to the specified arguments, defining the encoder and decoder
    configs.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        kwargs (*optional*):
            Dictionary of keyword arguments. Notably:

                - **encoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the encoder config.
                - **decoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the decoder config.

    Examples:
        TODO
    ```"""

    model_type = "LSTM2LSTM"
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert (
            "encoder" in kwargs and "decoder" in kwargs
        ), "Config has to be initialized with encoder and decoder config"
        encoder_config = kwargs.pop("encoder")
        decoder_config = kwargs.pop("decoder")

        self.encoder = encoder_config
        self.decoder = decoder_config

        self.decoder.is_decoder = True
        self.is_encoder_decoder = True

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default *to_dict()* from *PretrainedConfig*.

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["encoder"] = self.encoder.to_dict()
        output["decoder"] = self.decoder.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
