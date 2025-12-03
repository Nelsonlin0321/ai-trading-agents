# SandX AI Trading Agents

## Setup Python Environment
```shell
python -m venv .venv
source .venv/bin/activate
uv init
```


## Git Hooks
This repository includes a pre-commit hook that automatically clears outputs from Jupyter notebooks before committing. This ensures that:
- Notebook outputs are not stored in the git history
- Notebook files remain lightweight
- Output cells are re-executed fresh when the notebook is opened

The hook is automatically executed when you run `git commit` and will clear outputs from any `.ipynb` files you are committing.

For alternative setup using pre-commit framework, install it with:
```bash
uv add pre-commit
pre-commit install
```

```shell
source .env
source .venv/bin/activate
uv sync
# uv sync --group dev
```

```shell
python3 -m prisma generate
```

```shell
python main.py --bot_id 123
```
## Test
```shell
python -m test.src.services.alpaca.test_api_client
```


## Dev Env Config 
```shell
pylint --generate-rcfile > .pylintrc
```

## Learning Docs
- https://docs.langchain.com/oss/python/langchain/overview


```shell
npx codefetch --project-tree 5
```