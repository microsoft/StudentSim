# Data

Everything the three domains train and evaluate on lives under this directory.
Point `STUDENTSIM_DATA_DIR` elsewhere if you keep data outside the checkout.

```
data/
  chess/           downloaded, ready to use (see chess/README.md)
  l2/
    raw/           you put the EFCAMDAT extract here
    ...            build output
  math/
    raw/           you put the FoundationalASSIST extract here
    ...            build output
```

## Why the domains differ

Chess comes from the Lichess CC0 export, so the exact training and
evaluation files this pipeline consumes are redistributed. Nothing needs to be rebuilt
on your side.

EFCAMDAT and FoundationalASSIST both require an agreement with their providers,
so no records from them are redistributed. What ships instead is the raw-data
specification in each `raw/README.md` and one build script per domain that turns
that raw extract into the training and evaluation files the pipeline reads.
