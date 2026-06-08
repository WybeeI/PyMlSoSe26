# mnist_proj

Tiny MNIST classifier. Example project for PyML L2.

## Quickstart

```bash
uv venv
source .venv/bin/activate
uv pip install -e .

python -m scripts.train         # trains, writes runs/<timestamp>/
python -m scripts.predict       # loads latest run, predicts on a sample
pytest                          # runs tests
```

## Layout

```
mnist_proj/
  mnist_proj/        library code (importable, no side effects)
    data.py          dataset + transforms
    model.py         classifier architecture
    train.py         training loop (callable)
    eval.py          accuracy on a held-out split
  scripts/           entry points (CLI glue, __main__)
    train.py
    predict.py
  tests/             pytest tests
  runs/              run artifacts (created at first train)
  pyproject.toml     dependencies, package metadata
  AGENTS.md          guidance for AI agents working in this repo
```
