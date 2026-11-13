# Health-Forecast

Previsão de demanda mensal de procedimentos do Hospital das Clínicas da UFPE
(HC-UFPE), com o resultado adotado pela diretoria do HC na interlocução com o SUS.

Este repositório é a versão de portfólio: o notebook original virou um pacote
Python com dados sintéticos reprodutíveis, treino, avaliação e figuras.

**Versão 0.1** — ingestão reprodutível + treino e avaliação de um modelo por vez.

## O que tem

| Módulo | Faz |
|---|---|
| `sigma/dataset.py` | gera o dataset sintético HC-like (6 séries × 72 meses, seed 42) |
| `sigma/data_preparation.py` | ingere CSV long (`unique_id,ds,y`) ou wide DATASUS (`Jan/2018`, `Janeiro/2018`), com `,` ou `;`, UTF-8 ou latin-1, e descarta linhas/colunas `Total` |
| `sigma/models.py` | wrapper do StatsForecast com AutoARIMA, AutoETS, Theta, AutoCES, AutoTBATS e AutoTheta |
| `sigma/main.py` | CLI de treino, avaliação, previsão e gráfico |
| `sigma/utils.py` | métricas MAE, MSE, RMSE, MAPE e sMAPE (as percentuais ignoram pontos com demanda zero) |

## Começar

Requisitos: Python 3.12 e [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
make data      # data/raw/dataset.csv, 432 linhas sintéticas
make train     # treina, avalia, prevê e plota
```

`make train` escreve:

- `outputs/auto_arima_metrics.csv` — métricas por série
- `outputs/auto_arima_forecasts.csv` — previsão para o horizonte de 12 meses
- `outputs/plots/` — um PNG por série
- `models/auto_arima_model.pkl` + `.json` — modelo treinado e metadados

Com o venv ativo, a CLI aceita as opções direto:

```bash
python -m sigma.main --data-path data/raw/dataset.csv \
  --model auto_arima --train --evaluate --forecast --plot
```

Modelos disponíveis: `auto_arima`, `auto_ets`, `theta`, `auto_ces`, `auto_tbats`,
`auto_theta`. Sem `--train`, o comando carrega o modelo salvo em `models/`.

## Dados

O repositório não versiona dados do HC. `make data` gera uma amostra sintética
com sazonalidade 12, tendência e ruído, no formato `unique_id,ds,y` que a
pipeline espera. O notebook em `notebooks/` é o estudo original, versionado sem
outputs.

## Estrutura

```
sigma/            pacote (config, dataset, preparação, modelos, CLI)
tests/            pytest da ingestão, das métricas e da avaliação
notebooks/        estudo exploratório (sem outputs)
data/             gerado por `make data` (fora do git)
models/           modelos treinados (fora do git)
outputs/          métricas, previsões e figuras (fora do git)
```

## Qualidade

```bash
make lint   # ruff format --check + ruff check
make test   # pytest
```

## Próximo

- validação cruzada temporal com intervalos de 90% e análise de resíduos
- API FastAPI `/forecast` por `unique_id`
- Docker + `docker compose` para API e docs
- CI rodando `make lint` e `make test`

## Licença

MIT — ver [LICENSE](LICENSE).
