import os
import pickle
from typing import List, Dict, Any, Mapping

from lstm2lstm import LSTM2LSTMModel, LSTMConfig, LSTM2LSTMConfig, torch_lstm_data_collator
from transformers import AutoConfig
from rouge import Rouge
from transformers import BertTokenizer, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer
from dataset import PseudoCodeIterDatasetForLSTM2LSTMModel
from utils import compute_score_reassignment
from transformers import set_seed
from transformers.utils import logging
logging.enable_explicit_format()

set_seed(233)
AutoConfig.register("LSTM2LSTM", LSTM2LSTMConfig)
AutoConfig.register("LSTM", LSTMConfig)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
output_dir = "models/PC_Nero"
test_results_path = "results/PC_Nero_predictions.txt"
train_data = {
    "src": "data/PC/Nero/train/src-train-sorted.txt", # -sorted
    "tgt": "data/PC/Nero/train/tgt-train-sorted.txt", # -sorted
}
valid_data = {
    "src": "data/PC/Nero/valid/src-valid.txt",
    "tgt": "data/PC/Nero/valid/tgt-valid.txt",
}
test_data = {
    "src": "data/PC/Nero/test/src-test-len3000.txt",
    "tgt": "data/PC/Nero/test/tgt-test-len3000.txt",
}
EPOCHS = 10
max_steps = 50000
max_input_length = 1500
max_target_length = 30
train_batch_size = 32
test_batch_size = 32
learning_rate = 1e-3
beam_size = 10
summarization_pkl_path = "data/PC/summarization.txt.pkl"
tokenizer_src_vocab_file = "data/PC/Nero/vocabsrc.txt"
tokenizer_tgt_vocab_file = "data/PC/Nero/vocabtgt.txt"
# src_emb_file = "embeddings/word2vec.amd64.pctoken.128d.txt"
# tgt_emb_file = "embeddings/method_tokens.128d.txt"
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
rouge = Rouge()
with open(summarization_pkl_path, "rb") as f:
    summarization = pickle.load(f)
tgt_vocab = tokenizer_tgt.vocab.keys()

def compute_metrics_reassignment(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = tokenizer_tgt.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer_tgt.pad_token_id
    label_str = tokenizer_tgt.batch_decode(labels_ids, skip_special_tokens=True)
    result = compute_score_reassignment(pred_str, label_str, tgt_vocab, summarization)
    try:
        rouge_output = rouge.get_scores(hyps=pred_str, refs=label_str, avg=True, ignore_empty=True)["rouge-1"]
    except:
        print("[!] get rouge scores error!")
        rouge_output = {'p':0.0, 'r':0.0, 'f':0.0}
    return {
        "Prec.": result['p'],
        "Rec.": result['r'],
        "F1.": result['f'],
        "rouge-1-p": rouge_output['p'],
        "rouge-1-r": rouge_output['r'],
        "rouge-1-f": rouge_output['f'],
    }


# dataset
train_dataset = PseudoCodeIterDatasetForLSTM2LSTMModel(train_data['src'], train_data['tgt'],
                                      tokenizer_src_vocab_file, tokenizer_tgt_vocab_file)
valid_dataset = PseudoCodeIterDatasetForLSTM2LSTMModel(valid_data['src'], valid_data['src'],
                                      tokenizer_src_vocab_file, tokenizer_tgt_vocab_file)
test_dataset = PseudoCodeIterDatasetForLSTM2LSTMModel(test_data['src'], test_data['src'],
                                      tokenizer_src_vocab_file, tokenizer_tgt_vocab_file)
# Initializing a seq2seq model
encoder_config = LSTMConfig(
    vocab_size=tokenizer_src.vocab_size,
    tokenizer=tokenizer_src,
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
    tokenizer=tokenizer_tgt,
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
lstm2lstm_config = LSTM2LSTMConfig(encoder=encoder_config, decoder=decoder_config,
                                                                main_input_name="src",
                                                                )
model = LSTM2LSTMModel(lstm2lstm_config)

# set training arguments - these params are not really tuned, feel free to change
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    eval_steps=10000,
    # num_train_epochs=EPOCHS,
    max_steps=max_steps,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=test_batch_size,
    # auto_find_batch_size=True,
    gradient_accumulation_steps=1,
    warmup_steps=0,
    logging_dir="models/logs",
    logging_steps=1,
    save_strategy="steps",  # or "steps"
    save_steps=10000,
    # save_total_limit=3,
    local_rank=-1,
    dataloader_num_workers=0,  # must be 0 in win
    load_best_model_at_end=True,
    optim="adamw_torch",
    learning_rate=learning_rate,
    weight_decay=1.0,
    lr_scheduler_type='linear',
    generation_max_length=max_target_length,
    predict_with_generate=True,
    generation_num_beams=beam_size,
    report_to=["none"],
    run_name=os.path.basename(__file__),
)



# instantiate trainer
trainer = Seq2SeqTrainer( # num_beams
    model=model,
    tokenizer=tokenizer_tgt,
    args=training_args,
    data_collator=torch_lstm_data_collator,
    compute_metrics=compute_metrics_reassignment,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)
trainer.train()
trainer.evaluate()
# predict_results = trainer.predict(test_dataset=test_dataset)
# predictions = tokenizer_tgt.batch_decode(
#                     predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
#                 )
# trainer.log_metrics("test", predict_results.metrics)
# trainer.save_metrics("test", predict_results.metrics,combined=False)
# predictions = [pred.strip() for pred in predictions]
# with open(test_results_path, "w") as writer:
#     writer.write("\n".join(predictions))