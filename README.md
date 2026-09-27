# GPT from Scratch

A character-level GPT trained on Tiny Shakespeare, built step by step following Andrej Karpathy's "Let's build GPT": bigram → self-attention → multi-head attention → feed-forward → residual blocks → LayerNorm → dropout.

## Files

- `Bigram.py` — baseline bigram language model
- `gpt.py` — the full transformer model
- `gpt_from_scratch.ipynb` — notebook walking through the build
- `Residual Network .ipynb` — residual network experiments (uses `names.txt`)
- `input.txt` — Tiny Shakespeare dataset
- `names.txt` — names dataset

## Run

```bash
pip install torch matplotlib
python gpt.py
```

`matplotlib` is only needed for the notebooks.

## Results

| Stage | Iters / LR | Train @2700 | Val @2700 | Train @4800 | Val @4800 |
|---|---|---|---|---|---|
| Bigram (no attention) | 3000 / 1e-2 | 2.5040 | 2.5114 | – | – |
| One self-attention head | 3000 / 1e-2 | 2.4033 | 2.4352 | – | – |
| Multi-head attention | 3000 / 1e-2 | 2.2168 | 2.2792 | – | – |
| + Feed-forward | 3000 / 1e-2 | 2.1564 | 2.2431 | – | – |
| 4 residual blocks, no projection | 5000 / 1e-3 | 2.1103 | 2.1615 | 2.0196 | 2.0901 |
| 4 residual blocks, with projection | 5000 / 1e-3 | 2.0907 | 2.1384 | 1.9718 | 2.0651 |
| LayerNorm (pre-norm) | 5000 / 1e-3 | 2.0717 | 2.1273 | 1.9677 | 2.0710 |
| LayerNorm (post-norm) | 5000 / 1e-3 | 2.0782 | 2.1310 | 1.9742 | 2.0708 |
| Dropout 10% | 5000 / 1e-3 | 2.1308 | 2.1718 | 2.0522 | 2.1238 |
