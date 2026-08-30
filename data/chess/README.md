# Chess data

Source corpus: the Lichess May 2025 standard-time-control export, released CC0.
That licence lets the derived records be redistributed directly, so the training
and evaluation files are published separately from the code and nothing needs to
be rebuilt on your side. The upload has not landed yet.

## Download

The `<HF_ORG>/<HF_DATASET>` placeholder will be replaced with the real dataset
path when the upload lands.

```bash
huggingface-cli download <HF_ORG>/<HF_DATASET> \
    --repo-type dataset --local-dir data/chess
```

## What is here

| path | records | what it is |
|---|---:|---|
| `pooled/train.jsonl` | 100,000 | Stage-1 training across 100 students: 80,000 single-turn and 20,000 guidance records |
| `players/s035bc3bf.jsonl` … | 1,000 each | Stage-2 specialization, 30 students |
| `test_st/s035bc3bf.jsonl` … | 5,000 each | single-turn held-out records |
| `test_mt/s035bc3bf.jsonl` … | 4,000 each | multi-turn held-out records |

The names `test_st` and `test_mt` are the same names used in every domain in this repository, and `studentsim-eval` defaults to them. The same student ids are used across `players/`, `test_st/`, and `test_mt/`, so each student's training file and held-out files share a name.

Student ids are random and carry no relation to accounts on the source site. No mapping back to source-site accounts exists.

A record is a chat. A single-turn record is `[user, assistant]`: the position,
and the move the student played. A guidance record is
`[user, assistant, user, assistant]`, where the second user turn is the tutor
and the last assistant turn is what the student did after reading it.

## Per-student files

The files under `players/` are already the Stage-2 training subset and can be trained on directly.

Each file contains 800 single-turn records plus 50 records for each of the four guidance modes. The equal counts per mode make the per-mode responsiveness breakdown comparable across modes.

The four guidance modes are `strategic`, `socratic`, `error_remediation`, and `comparative`. Multi-turn records carry the mode in an `instruction_type` field. Each `test_mt` file has 1,000 records per mode.
