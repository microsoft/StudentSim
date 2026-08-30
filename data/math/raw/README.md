# Put the FoundationalASSIST extract here

The mathematics domain is built from FoundationalASSIST, which ASSISTments distributes behind a Responsible Use Agreement. That agreement licenses the corpus to the user rather than making it redistributable, so this directory starts empty and you fill it from your own copy.

## What to obtain

Accept the agreement on the dataset page, then download the three CSVs into this directory:

```text
data/math/raw/Interactions.csv
data/math/raw/Problems.csv
data/math/raw/Skills.csv
```

## Then build

Two steps call a language model, so configure credentials for your provider first.

Some answer keys in the release are wrong, through arithmetic slips and through problems whose figure was lost when the text was extracted, leaving a question that cannot be answered from what remains. The first step has a model solve each problem on its own and marks the keys it disputes, so that no student is scored against an answer that is not the right one. Run it once and reuse the result.

```bash
python -m studentsim.data.math.audit \
    --raw data/math/raw --out data/math/audit_excluded.json

python -m studentsim.data.math.build \
    --raw data/math/raw --out data/math --exclude data/math/audit_excluded.json
```

The build refuses to run without an audit list rather than quietly treating every key as correct. `--no-audit` builds anyway if that is what you want; the resulting corpus differs from the one the released configs assume.

The second step writes the guidance the tutor gives. By default it generates all guidance styles in one run, writing one file per style under the output directory; `--style` is optional and repeatable if you want only a subset:

```bash
python -m studentsim.data.math.generate
```

This step exists because the source corpus is licensed to the user rather than redistributable, so neither the source records nor anything derived from them can be shipped here. When it finishes, the generated records follow the StudentSimEval protocol for the same problem, the same student answer, and the same correct answer, but with tutor guidance written by the model. The wording therefore differs from run to run.
