# Notebook 26 – Isolated environment

`langgraph-codeact` requires `langgraph>=0.3.5,<0.4.0`, while the main project pins
`langgraph>=1.1.6`. This sub-project keeps an isolated venv just for
`26_CodeAgent_Analisis_Inteligente_GithubIssues.ipynb`.

## Setup (already done)

```bash
cd envs/notebook26
uv sync --python 3.12
.venv/bin/python -m ipykernel install --user \
    --name notebook26-codeact \
    --display-name "Python 3.12 (notebook26 codeact)"
```

## How to use

1. Open `26_CodeAgent_Analisis_Inteligente_GithubIssues.ipynb`.
2. In the kernel picker, choose **Python 3.12 (notebook26 codeact)**.
3. Remove or comment the `!pip install ...` cell — packages are already installed.

## Versions

- python 3.12
- langgraph 0.3.x
- langgraph-codeact 0.1.x
- langchain 0.3.x
- langchain-anthropic 0.3.x

## Environment variables

Loaded via `python-dotenv` from the root `.env`. Make sure
`ANTHROPIC_API_KEY` is set there.

## Uninstall

```bash
jupyter kernelspec uninstall notebook26-codeact
rm -rf envs/notebook26/.venv envs/notebook26/uv.lock
```
