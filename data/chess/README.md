# Chess data

Source corpus: the Lichess May 2025 standard-time-control export, released CC0.
That licence lets the derived records be redistributed, so the training and
evaluation files are released as they are and nothing needs to be rebuilt on your side.

## Download

```bash
huggingface-cli download <HF_ORG>/<HF_DATASET> \
    --repo-type dataset --local-dir data/chess
```

## What is here

| path | records | what it is |
|---|---:|---|
| `pooled/train.jsonl` | 100,000 | Stage-1 training across 100 students: 80,000 single-turn and 20,000 guidance records |
| `players/p00.jsonl` … `players/p29.jsonl` | 1,000 each | Stage-2 specialization, 30 students |
| `test_st/p00.jsonl` … `test_st/p29.jsonl` | 5,000 each | single-turn held-out records |
| `test_mt/p00.jsonl` … `test_mt/p29.jsonl` | 4,000 each | multi-turn held-out records |

The names `test_st` and `test_mt` are the same names used in every domain in this repository, and `studentsim-eval` defaults to them. The student ids `p00` through `p29` match across `players/`, `test_st/`, and `test_mt/`, so each student's training file and held-out files share a name.

A record is a chat. A single-turn record is `[user, assistant]`: the position,
and the move the student played. A guidance record is
`[user, assistant, user, assistant]`, where the second user turn is the tutor
and the last assistant turn is what the student did after reading it.

## Per-student files

The files under `players/` are already the Stage-2 training subset and can be trained on directly.

The released per-student files are the draw at data sampler seed 42. That seed is already the default in `configs/training/stage1_chess.yaml` and `configs/training/stage2_chess.yaml`. Running `python -m studentsim.data.chess.subsample` at that seed reproduces the released per-student files exactly, so the shipped configs and the shipped data agree and nothing needs to be redrawn.

Each file contains 800 single-turn records plus 50 records for each of the four guidance modes. The equal counts per mode make the per-mode responsiveness breakdown comparable across modes.

The four guidance modes are `strategic`, `socratic`, `error_remediation`, and `comparative`. Multi-turn records carry the mode in an `instruction_type` field. Each `test_mt` file has 1,000 records per mode.
