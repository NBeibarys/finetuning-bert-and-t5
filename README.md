# Fine-tuning BERT and T5

NYU DS-GA 1011 (Natural Language Processing with Representation Learning), Assignment 4. I fine-tuned BERT for IMDB sentiment classification, designed a full out-of-distribution data transformation and augmentation study around it, then fine-tuned T5-small to generate SQL from natural language, evaluated by executing the generated SQL on a real flight database.

## Results

| Task | Model | Result |
| --- | --- | --- |
| IMDB sentiment (25k test) | BERT fine-tuned 3 epochs | 93.22% accuracy |
| Same model on transformed test set | lowercase + slang + sarcasm + emoji + typos | 87.78% (5.44 point drop) |
| Augmented model on transformed test set | + 5k transformed training copies | 96.55% |
| Augmented model on original test set |  | 92.58% |
| Text-to-SQL, dev set | T5-small, lr 3e-4, beam 4 | 0.69 Record F1, 0.64 Record EM |

## Part 1: BERT sentiment with an OOD augmentation study

The training loop is written from scratch on top of the starter skeleton: batching, device placement, AdamW with a linear LR schedule, early evaluation. The interesting half is the data transformation I designed: it makes a review read like it was typed fast on a phone in 2026 - all lowercase, slang substitutions ("very good" becomes "fire"), a sarcastic closing line drawn from label-dependent pools, one sentiment emoji, and QWERTY-neighbor typos. Every part is deterministic and reproducible from the writeup.

The augmentation study found something I did not expect: the augmented model scores *higher* on the transformed test set (96.55%) than on the clean one (92.58%). The cause is label leakage - my sarcastic closing lines are drawn from pools keyed by the review's label, so every transformed example ends with a sentence whose wording reveals the answer, and the model learned to read it. Part of the improvement is real robustness, part is the model exploiting an artifact of how the transformation was built. The full analysis with all four accuracy numbers is in the report.

## Part 2: T5-small text-to-SQL

Encoder-decoder conditional generation: a natural language flight query goes in, SQL comes out. Implementation choices:

- Default T5 tokenizer on both sides, end-of-sentence token preserved on SQL targets as the stop signal
- A spare token (`<extra_id_0>`) appointed as begin-of-sentence, used identically in training and generation
- Teacher forcing: decoder input = BOS + target[:-1], so each position predicts the next token
- Dynamic padding per batch, no truncation (a truncated SQL query is a broken query)
- Fine-tuning the full model, AdamW at lr 3e-4, cosine schedule with 1 warmup epoch, early stopping on dev Record F1
- Beam search (num_beams=4) at generation, capped at 512 tokens

The learning-rate run is the part I would keep: at 1e-4 the model plateaued at 0.447 dev F1; at 3e-4 the same setup reached 0.669 greedy and 0.692 with beam search. Evaluation executes every generated query against the SQLite database and compares returned records against the ground-truth queries' records, which is the right metric since many different SQL queries answer the same question.

Error analysis on dev (466 queries): 56% reproduce the gold records exactly, and the failures split into hallucinated schema names, malformed time conditions with repetition loops, incomplete generation on long multi-join queries, and valid-but-wrong SQL - all with counts and examples in the report.

## Reproducing

Each part pins different package versions, so use a separate environment for each. On a Blackwell GPU the pinned torch 2.1.2 has no kernels and must be upgraded to a cu128 build after installing requirements.

```bash
# Part 1
conda create -n dsga1011-hw4-part1 python=3.11
conda activate dsga1011-hw4-part1
cd part1 && pip install -r requirements.txt
python main.py --train --eval              # trains on full IMDB, writes out_original.txt

# Part 2
conda create -n dsga1011-hw4-part2 python=3.11
conda activate dsga1011-hw4-part2
cd part2 && pip install -r requirements.txt
python train_t5.py --finetune --learning_rate 3e-4 --weight_decay 0 \
    --scheduler_type cosine --num_warmup_epochs 1 --max_n_epochs 20 \
    --patience_epochs 3 --batch_size 16 --test_batch_size 16 \
    --experiment_name baseline_v2_lr3e4
```

Course assignment by Greg Durrett, NYU DS-GA 1011, Fall 2026. Starter skeleton by the course staff; all filled-in code is mine, written with help from my own AI (GLM 5.3), which the assignment permits - I describe exactly where I used it in the writeup.
