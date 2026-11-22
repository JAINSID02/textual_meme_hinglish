# Hinglish Meme Classifier

A small transformer (~10M parameters), written from scratch in PyTorch, that sorts Hinglish
meme text into categories. It is pretrained on unlabeled Hinglish text, then fine-tuned with a
classification head. No Hugging Face model classes or Trainer; the tokenizer uses the
`tokenizers` library for BPE training only.

## Pipeline

raw text -> clean -> BPE tokenizer -> Hinglish pretraining -> classifier fine-tuning -> evaluation -> demo

## Repository structure

meme_llm/
  config.py                 categories + all hyperparameters
  data/clean.py             text cleaning (keeps emojis and Devanagari)
  data/split.py             label check, dedupe, stratified train/val/test split
  tokenizer/train_tokenizer.py   BPE training + tokenizer wrapper
  model/layers.py           RMSNorm, SwiGLU
  model/rope.py             rotary position embeddings
  model/attention.py       causal self-attention
  model/transformer.py     backbone + language-model head
  model/classifier.py      backbone + masked mean pooling + class head
  training/lm_dataset.py   corpus -> non-overlapping blocks (1 pass = 1 epoch)
  training/utils.py        LR schedule, optimizer, atomic checkpoints
  training/pretrain.py     Hinglish pretraining loop
  training/cls_dataset.py  padded batches + class weights
  training/finetune_cls.py classifier training, early stopping on macro-F1
  eval/metrics.py          test-set metrics, confusion matrix, error list
  eval/baseline.py         TF-IDF + logistic regression baseline
  eval/predict.py          inference entry point
  app/app.py               Streamlit demo

## Model

| Setting | Value |
|---|---|
| Layers / heads / width | 6 / 8 / 384 |
| Context length | 256 (classifier uses max 128) |
| Vocabulary | 8,000 byte-level BPE (trained on Hinglish) |
| Position encoding | RoPE |
| Normalization / FFN | RMSNorm / SwiGLU |
| Pooling | mean over non-pad tokens |

## Data

- Pretraining corpus: TODO (source, size in MB, number of tokens)
- Labeled memes: TODO (source, number of examples, how labels were created and checked)
- Categories: TODO (list, with the one-line definition of each)
- Split: stratified 80/10/10, exact duplicates removed before splitting

## How to run

pip install -r requirements.txt
cd meme_llm

# 1. prepare data (put files in data/raw/ first)
python -c "from data.clean import clean_labeled_csv, clean_corpus; clean_labeled_csv(); clean_corpus()"
python data/split.py

# 2. tokenizer and pretraining
python tokenizer/train_tokenizer.py
python -m training.pretrain          # add --resume to continue

# 3. classifier
python -m training.finetune_cls
python -m training.finetune_cls --from_scratch --out runs/classifier_scratch   # ablation

# 4. evaluation and demo
python -m eval.metrics
python -m eval.baseline
streamlit run app/app.py

## Results (held-out test set)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + logistic regression | TODO | TODO |
| Transformer, no pretraining | TODO | TODO |
| Transformer, Hinglish pretraining | TODO | TODO |

Per-class scores and the confusion matrix are in runs/classifier/.
TODO: one or two sentences on what the comparison shows, stated plainly,
including if the baseline is competitive.

## Limitations

- Text only: memes that depend on the image are classified from their caption alone.
- Sarcasm and overlapping categories make some labels subjective; see test_errors.csv for examples.
- Small model and small labeled set: per-class scores for rare categories are noisy.
- Confidence scores are not calibrated; the demo shows "not sure" below a threshold.
- Scraped text may contain offensive language; review it before publishing a dataset or demo.