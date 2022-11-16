# ref: https://github.com/shushanxingzhe/transformers_ner/blob/main/models.py
# ref: https://huggingface.co/docs/transformers/custom_models
import torch
from torch import nn
from transformers.modeling_outputs import Seq2SeqLMOutput

from lstmconfig import LSTMConfig, LSTM2LSTMConfig
from transformers import AutoConfig, PretrainedConfig
from typing import List, Tuple, Any, Dict
from transformers import PreTrainedModel
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pad_sequence as pad
from global_attention import GlobalAttention
from gate import context_gate_factory


def torch_lstm_data_collator(features: List[Dict]) -> Dict[str, Any]:
    import torch
    first = features[0]
    batch = {}
    for k in first.keys():
        batch[k] = torch.stack([f[k] for f in features])

    # handle data for lstm2lstm model
    max_src_len = batch['src_lengths'].max()
    batch['src'] = batch['src'][:,:max_src_len].transpose(0, 1)
    max_tgt_len = batch['tgt_lengths'].max()
    batch['tgt'] = batch['tgt'][:, :max_tgt_len].transpose(0, 1)
    return batch


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    # TODO copy from transformers, may need modified
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class LSTMEncoder(PreTrainedModel):
    config_class = LSTMConfig

    def __init__(self, config: LSTMConfig, **kwargs):
        super().__init__(config)
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.input_size)
        self.enc_lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            bias=config.bias,
            batch_first=config.batch_first,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
        )
        self.dropout = nn.Dropout(config.dropout)

        self.main_input_name = kwargs.pop("main_input_name", "src")

    def forward(self, src:torch.Tensor, lengths=None, **kwargs):
        # src: (src_len,batch_size)
        # lengths: (batch_size,)

        if lengths is None:
            _lengths = []
            for s in src.transpose(0,1):
                find_idx = (s==self.config.pad_token_id).nonzero()
                if len(find_idx) == 0:
                    _lengths.append(len(s))
                else:
                    _lengths.append(int(find_idx[0]))
            lengths = torch.tensor(_lengths)

        embs = self.embeddings(src)

        packed_embs = pack(embs, lengths.view(-1).tolist(), enforce_sorted=False) if lengths is not None else embs

        enc_output, enc_state = self.enc_lstm(packed_embs)

        enc_output = unpack(enc_output)[0] if lengths is not None else enc_output

        return enc_output, enc_state, lengths


class LSTMDecoder(PreTrainedModel):
    config_class = LSTMConfig

    def __init__(self, config: LSTMConfig, **kwargs):
        super().__init__(config)
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.input_size)
        self.dec_lstm = nn.LSTM(
            input_size=config.input_size+config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            bias=config.bias,
            batch_first=config.batch_first,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
        )

        if config.attn_type:
            if config.attn_type == "general":
                self.attn = GlobalAttention(config.hidden_size, coverage=False,
                    attn_type=config.attn_type, attn_func="softmax")
            else:
                raise ValueError(f"unknown attention type: {config.attn_type}")

        if config.context_gate is not None:
            self.context_gate = context_gate_factory(
                config.context_gate, config.input_size,
                config.hidden_size, config.hidden_size, config.hidden_size
            )

        self.dropout = nn.Dropout(config.dropout)

        self.state = {}


    def forward(self, tgt, enc_output, src_lengths=None):
        # tgt: (tgt_len,batch_size)
        # enc_output: (src_len,batch_size,enc_hidden_size*(bidirectional+1))
        # src_lengths: (batch_size,)

        embs = self.embeddings(tgt)
        assert embs.dim() == 3  # tgt_len x batch_size x embedding_dim

        dec_state = (self.state["hidden"], self.state["cell"])
        input_feed = self.state["input_feed"]

        dec_outputs = []
        attns = {"std": []}
        for emb in embs.split(1):
            dec_input = torch.cat([emb, input_feed], -1)
            lstm_output, dec_state = self.dec_lstm(dec_input, dec_state)
            if self.config.attn_type is not None:
                dec_output, p_attn = self.attn(
                    lstm_output.transpose(0, 1),
                    enc_output.transpose(0, 1),
                    memory_lengths=src_lengths)
                attns["std"].append(p_attn)
            else:
                dec_output = lstm_output
            if self.config.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                dec_output = self.context_gate(
                    dec_input, lstm_output, dec_output
                )
            dec_output = self.dropout(dec_output)
            input_feed = dec_output
            dec_outputs.append(dec_output.squeeze(0))

        # Update the state with the result.
        self.state["hidden"] = dec_state[0]
        self.state["cell"] = dec_state[1]
        self.state["input_feed"] = dec_outputs[-1]

        # Concatenates sequence of tensors along a new dimension.
        dec_outputs = torch.stack(dec_outputs)
        for k in attns:
            if isinstance(attns[k], List):
                attns[k] = torch.stack(attns[k])

        return dec_outputs, attns

    def init_state(self, enc_state: Tuple[torch.Tensor, torch.Tensor]):
        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.config.bidirectional_encoder:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)
            return hidden

        self.state["hidden"] = _fix_enc_hidden(enc_state[0])
        self.state["cell"] = _fix_enc_hidden(enc_state[1])

        # Init the input feed.
        batch_size = self.state["hidden"].size(1)
        h_size = (batch_size, self.config.hidden_size)
        self.state["input_feed"] = self.state["hidden"].data.new(*h_size).zero_().unsqueeze(0)

    def _reorder_cache(self, past, beam_idx):
        raise NotImplementedError


class LSTM2LSTMModel(PreTrainedModel):
    config_class = LSTM2LSTMConfig

    def __init__(self, config: PretrainedConfig, **kwargs):
        super(LSTM2LSTMModel, self).__init__(config)

        self.encoder = LSTMEncoder(config.encoder)
        self.decoder = LSTMDecoder(config.decoder)
        self.generator = nn.Sequential(
                nn.Linear(config.decoder.hidden_size, config.decoder.vocab_size),
                nn.LogSoftmax(dim=-1)
            )
        decoder_pad_token_id = config.decoder.pad_token_id if config.decoder.pad_token_id else -100
        self.loss_function = nn.NLLLoss(ignore_index=decoder_pad_token_id, reduction='sum')

        self.main_input_name = kwargs.pop("main_input_name", "src")

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_encoder_decoder_pretrained(
            cls,
            encoder_pretrained_model_name_or_path: str = None,
            decoder_pretrained_model_name_or_path: str = None,
            *model_args,
            **kwargs
    ) -> PreTrainedModel:
        raise NotImplementedError

    def forward(self, src, tgt, src_lengths, tgt_lengths, **kwargs):
        # src: (src_len,batch_size)
        # tgt: (tgt_len,batch_size)
        # src_lengths: (batch_size,)
        # tgt_lengths: (batch_size,)

        # # test speed
        # import pickle
        # with open("./data/speedtest.data.pkl","rb") as f:
        #     data = pickle.load(f)
        # src = data['src'].squeeze(-1)
        # tgt = data['tgt'].squeeze(-1)
        # src_lengths = data['lengths']

        # import time
        # start_time = time.time()
        # print("\n[-] max src len: ", src_lengths.max())

        dec_input = tgt[:-1]  # exclude last target from inputs

        enc_output, enc_state, _lengths = self.encoder(src, src_lengths)

        self.decoder.init_state(enc_state)
        dec_output, _attns = self.decoder(dec_input, enc_output, src_lengths)

        logits = self.generator(dec_output)

        scores = logits.view(-1, logits.size(-1))
        gtruth = tgt[1:].contiguous().view(-1)
        loss = self.loss_function(scores, gtruth)

        # end_time = time.time()
        # print("[-] time cost loss func:  {}".format(end_time-start_time))

        return Seq2SeqLMOutput(loss=loss)


    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        raise NotImplementedError

    def prepare_inputs_for_generation(
            self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # TODO impl
        raise NotImplementedError
    def _reorder_cache(self, past, beam_idx):
        return self.decoder._reorder_cache(past, beam_idx)


if __name__ == '__main__':
    from transformers import BertTokenizer
    from lstmconfig import LSTM2LSTMConfig
    AutoConfig.register("LSTM2LSTM", LSTM2LSTMConfig)
    AutoConfig.register("LSTM", LSTMConfig)

    tokenizer_src_vocab_file = "data/PC/Nero/vocabsrc.txt"
    tokenizer_tgt_vocab_file = "data/PC/Nero/vocabtgt.txt"
    tokenizer_src = BertTokenizer(
        tokenizer_src_vocab_file, unk_token="<unk>",
        sep_token="<sep>", pad_token="<pad>", cls_token="<cls>",
        mask_token="<mask>", bos_token="<s>", eos_token="</s>"
    )
    tokenizer_tgt = BertTokenizer(
        tokenizer_tgt_vocab_file, unk_token="<unk>",
        sep_token="<sep>", pad_token="<pad>", cls_token="<cls>",
        mask_token="<mask>", bos_token="<s>", eos_token="</s>"
    )

    encoder_config = LSTMConfig(
        vocab_size=tokenizer_src.vocab_size,
        input_size=128,
        hidden_size=128,
        num_layers=2,
        bias=True,
        batch_first=False,
        dropout=0.5,
        bidirectional=True,
    )
    decoder_config = LSTMConfig(
        vocab_size=tokenizer_tgt.vocab_size,
        input_size=128,
        hidden_size=256,
        num_layers=2,
        bias=True,
        batch_first=False,
        dropout=0.5,
        bidirectional=False,
        is_decoder=True,
        bidirectional_encoder=True,
        attn_type="general",
    )

    lstm2lstm_config = LSTM2LSTMConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
    model = LSTM2LSTMModel(lstm2lstm_config)
    print(model)

    src_text  = "void __fastcall no return star ( ( __int64 a1 , __int64 a2 , void ( * a3 ) ( void ) ) { __int64 v3 ; int v4 ; __int64 v5 ; char * ret addr ; v4 = v5 ; v5 = v3 ; lib c sub main ( main , v4 , & ret addr , init , fini , a3 , & v5 ) ; halt ( ) ; }"
    tgt_text = "func start main"
    src = tokenizer_src([src_text,src_text,src_text], max_length=1500,
                                truncation=True, return_tensors='pt', return_length=True)
    with tokenizer_tgt.as_target_tokenizer():
        tgt = tokenizer_tgt([tgt_text,tgt_text,tgt_text], max_length=30,
                                    truncation=True, return_tensors='pt', return_length=True)
    lengths = src['length']
    src = src['input_ids'].transpose(0,1)
    tgt = tgt['input_ids'].transpose(0,1)
    prediction = model(src,tgt,lengths)
    scores = prediction.view(-1, prediction.size(2))
    gtruth = tgt[1:].contiguous().view(-1)

    padding_idx = 1
    criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')
    loss = criterion(scores, gtruth)
    loss.backward()
