# Mimex

**Study, train, and run an AI text detector on your own machine.**

Mimex is an open-source Python project for exploring AI-generated text detection.
It includes a browser interface, command-line predictions, a local HTTP API, and
training and evaluation workflows built around ModernBERT.

Use it to learn how a text classifier works, reproduce detection experiments,
or build on the code for your own research.

## What’s included

- A local browser interface for analyzing English text.
- CLI commands for training, evaluation, and prediction.
- Versioned preprocessing and overlapping-window inference for full documents.
- Separate training, calibration, and evaluation splits with overlap checks.
- Cross-dataset experiments, subgroup results, and documented failure cases.

## Quickstart

You’ll need **Python 3.12** and [uv](https://docs.astral.sh/uv/).
Apple Silicon, CUDA, and CPU execution are supported through PyTorch; available
hardware is selected automatically. The original experiments ran on an M4 Pro
with 24 GB memory. CPU training will take longer.

```sh
git clone https://github.com/vaibhav0806/mimex.git
cd mimex
uv sync --frozen
uv run mimex doctor
```

**Train a local baseline.** The repository includes training code, not pretrained
Mimex weights. These commands download the public HC3 data and base model, prepare
separate data splits, and train the original baseline:

```sh
uv run mimex prepare
uv run mimex train
```

**Open the app:**

```sh
uv run mimex serve --checkpoint checkpoints/hc3
```

Visit **http://127.0.0.1:7860** and paste at least 50 words of English text.
Inference runs locally. Initial installation and data/model preparation require
internet access.

The quickstart builds the original HC3 baseline, which analyzes the first 256
model tokens. For full-document models and the v2/v3 experiments, follow the
[reproduction guide](REPRODUCING.md).

## Use the CLI

Analyze a text file or read from stdin:

```sh
uv run mimex predict --checkpoint checkpoints/hc3 --file article.txt
cat article.txt | uv run mimex predict --checkpoint checkpoints/hc3
```

Evaluate the baseline on its held-out split:

```sh
uv run mimex evaluate --checkpoint checkpoints/hc3
```

Use a different locally trained checkpoint:

```sh
uv run mimex serve --checkpoint checkpoints/v3 --port 7860
uv run mimex predict --checkpoint checkpoints/v3 --file article.txt
```

Discover available commands with `uv run mimex --help` or add `--help` to any
subcommand. Preparation and training refuse to overwrite populated destinations;
choose a new data or checkpoint path for another experiment.

## Local API

The app serves an HTTP API on `127.0.0.1`:

| Endpoint | Purpose |
|---|---|
| `GET /` | Browser interface |
| `GET /health` | Model scope and scoring mode |
| `POST /predict` | Analyze a JSON object with a `text` string |

Send `Content-Type: application/json`. Inputs require at least 50 words, and
request bodies are limited to 100,000 bytes. The server is intended for local
use and processes requests sequentially.

## Study the implementation

| File or directory | What to explore |
|---|---|
| [data.py](src/mimex/data.py) | Data preparation, normalization, and split verification |
| [detector.py](src/mimex/detector.py) | Fine-tuning, threshold calibration, and document scoring |
| [metrics.py](src/mimex/metrics.py) | Detection rates, human false positives, and confidence intervals |
| [server.py](src/mimex/server.py) | Browser interface and local API |
| [cli.py](src/mimex/cli.py) | Command-line entry points |
| [scripts/](scripts/) | Evaluation workflows |
| [tests/](tests/) | Small tests using toy models, without trained weights |

Data, checkpoints, and run outputs are saved under `data/`, `checkpoints/`, and
`runs/`. These directories are ignored by Git.

## Research and limitations

Mimex is an experimental detector. Performance varies substantially across
writing styles, datasets, and generators. A score is not an authorship
probability, and a result cannot establish who wrote a passage. Mixed authorship
and non-English text have not been validated for general use.

The reports include both successful results and failures, including substantial
human false positives in some writing categories:

- [Original baseline model card](MODEL_CARD.md)
- [V2 results](EVALUATION.md) and [experiment protocol](EXPERIMENT.md)
- [V3 results](EVALUATION_V3.md) and [experiment protocol](EXPERIMENT_V3.md)
- [Additional external tests](EVALUATION_EXTERNAL_V3.md) and [evaluation protocol](EXTERNAL_V3_PROTOCOL.md)
- [Training and evaluation reproduction guide](REPRODUCING.md)

## Development and contributions

```sh
uv sync --frozen
uv run python -m unittest discover -s tests -v
uv run ruff check .
uv run ruff format --check .
```

Contributions to data quality, evaluation, documentation, and model behavior are
welcome. For experiment changes, document data provenance, keep related source
texts together across splits, and report human false positives alongside AI
recall. Use fresh holdouts for model selection.

## License

Code is available under the [MIT License](LICENSE). Models and datasets retain
their upstream licenses; see [data and model attribution](REPRODUCING.md#licenses-and-attribution).
The repository does not redistribute datasets or trained Mimex weights.
