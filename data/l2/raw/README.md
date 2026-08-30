# Put the EFCAMDAT extract here

The second-language English writing domain is built from EFCAMDAT, which is
distributed under an academic user agreement. That agreement licenses the
corpus to the user rather than making it redistributable, so this directory
starts empty and you fill it from your own copy.

## What to obtain

The **cleaned error-coded subcorpus** of EFCAMDAT (Öksüz et al., 2025). This is
the release that carries span-level teacher corrections, which the multi-turn
records need. The other EFCAMDAT releases do not carry them and will not work.

Request access from the maintainers of the corpus. The
build reads one file:

```
data/l2/raw/ef_POStagged_original_corrected.csv
```

## Then build

```bash
python -m studentsim.data.l2.build
```
