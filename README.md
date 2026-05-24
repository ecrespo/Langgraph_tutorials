# LangGraph Tutorials

Colección práctica de **notebooks de Jupyter** que ilustran patrones de agentes con
[LangGraph](https://langchain-ai.github.io/langgraph/) usando **Claude (Anthropic)** como LLM.
Van desde los fundamentos (estado tipado, checkpoints, branching, paralelismo) hasta
arquitecturas multi-agente, RAG agéntico, Computer Use Agents (CUA) y Human-in-the-Loop (HITL).

Todo está pensado para **usarse desde cero**: clonas el repo, instalas dependencias, creas
tu `.env`, descargas la data de cada notebook y ejecutas las celdas de arriba abajo.

> La integración LLM está unificada en **Anthropic Claude**. La API key se carga vía
> `python-dotenv` desde `.env`.

---

## Tabla de contenidos

- [Requisitos](#requisitos)
- [Instalación desde cero](#instalación-desde-cero)
- [Configuración del `.env`](#configuración-del-env)
- [Catálogo de notebooks](#catálogo-de-notebooks)
- [Datos necesarios](#datos-necesarios)
- [Cómo ejecutar los notebooks](#cómo-ejecutar-los-notebooks)
- [Entorno aislado para el notebook 26](#entorno-aislado-para-el-notebook-26)
- [Posibles problemas (issues) y soluciones](#posibles-problemas-issues-y-soluciones)
- [Modelos Claude disponibles](#modelos-claude-disponibles)
- [Estructura del repo](#estructura-del-repo)

---

## Requisitos

- **Python ≥ 3.13** para el entorno principal (notebooks `01`–`25`, `27`–`29`).
- **Python 3.12** para el entorno aislado del notebook `26` (lo crea `uv` automáticamente).
- [uv](https://docs.astral.sh/uv/) como gestor de paquetes (recomendado).
- Una **API key de Anthropic** (https://console.anthropic.com/).
- Opcional según el notebook: credenciales de **Kaggle** y/o un **token de Hugging Face**
  para descargar datasets (ver [Datos necesarios](#datos-necesarios)).

---

## Instalación desde cero

```bash
# 1. Clonar el repositorio
git clone <repo-url> && cd Langgraph_tutorials

# 2. Crear el entorno e instalar TODAS las dependencias del pyproject.toml
uv sync

# 3. Copiar la plantilla de variables de entorno y completarla
cp .env.example .env
$EDITOR .env        # pon tu ANTHROPIC_API_KEY (ver siguiente sección)
```

`uv sync` crea `.venv/` (Python 3.13) e instala LangChain, LangGraph, `langchain-anthropic`,
pandas, matplotlib, ChromaDB, sentence-transformers, Playwright, etc., según `pyproject.toml`
y el `uv.lock` fijado.

### Alternativa sin uv (pip)

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .          # usa pyproject.toml
# o, si prefieres, instala el stack mínimo:
# pip install langchain langchain-anthropic langgraph langchain-core \
#             python-dotenv pandas matplotlib jupyterlab
```

### Registrar el kernel de Jupyter (opcional pero recomendado)

```bash
uv run python -m ipykernel install --user \
    --name langgraph-tutorials --display-name "Python (langgraph-tutorials)"
```

Luego, en cada notebook, selecciona ese kernel. El **notebook 26 usa un kernel distinto**
(ver su sección).

---

## Configuración del `.env`

`.env.example` trae la plantilla. Copia a `.env` y completa al menos `ANTHROPIC_API_KEY`:

```env
# Anthropic Claude API (obligatorio)
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Modelo por defecto (configurable por notebook)
# Opciones: claude-opus-4-7, claude-sonnet-4-6, claude-haiku-4-5-20251001
ANTHROPIC_MODEL=claude-sonnet-4-6

# Opcionales según el notebook
TAVILY_API_KEY=tvly-dev-your-tavily-key-here   # búsqueda web (si aplica)
HF_TOKEN=hf_your_hf_token_here                 # datasets de Hugging Face
```

`python-dotenv` carga el `.env` automáticamente (`load_dotenv()` ya está en los notebooks).

> **Kaggle**: las credenciales NO van en el `.env`. Coloca tu `kaggle.json` en
> `~/.kaggle/kaggle.json` (`chmod 600`) — lo obtienes en *Kaggle → Account → Create New API Token*.

---

## Catálogo de notebooks

Todos los ejemplos son ahora **notebooks `.ipynb`**. Los notebooks `07`–`16` se generaron a
partir de paquetes Python; el código fuente original de esos módulos sigue disponible en sus
carpetas homónimas (`07_*/`, …, `16_*/`) como referencia, pero la forma recomendada de
ejecutarlos es el notebook.

Cada notebook incluye, tras el título, una celda **"¿Qué hace este notebook?"** y otra
**"Ejemplo de uso"** que documenta los datos de entrada y, en los flujos HITL, el
`Command(resume=...)` que el agente espera para continuar.

### Fundamentos (01–06)

| # | Notebook | Tema |
|---|---|---|
| 01 | `01_langgraph_state_typed.ipynb` | StateGraph con estado tipado (`TypedDict`) |
| 02 | `02_managing_graph_state.ipynb` | Reducers (`Annotated[Sequence, operator.add]`) y acumulación de estado |
| 03 | `03_persistent_checkpoint.ipynb` | Checkpointing con SQLite: resume, time-travel, branching |
| 04 | `04_multibranch_execution.ipynb` | Enrutamiento multi-rama + reanudación desde checkpoint |
| 05 | `05_conditional_router.ipynb` | Routing condicional con `add_conditional_edges` |
| 06 | `06_parallel_execution_and_subgraph.ipynb` | Ejecución en paralelo + invocación de subgrafos |

### Patrones intermedios (07–16)

| # | Notebook | Patrón |
|---|---|---|
| 07 | `07_implementing_human_in_the_loop.ipynb` | HITL básico con `interrupt()` (resumen de incidentes) |
| 08 | `08_Designing_a_Multi-Stage_Approval_Workflow.ipynb` | Aprobación multi-etapa (code review → testing → manager → UX) |
| 09 | `09_Using_Execution_Log_and_Resuming_from_Checkpoints.ipynb` | Log de ejecución + reanudar desde un `checkpoint_id` |
| 10 | `10_Performing_Branch_Analysis_with_Snapshots.ipynb` | Branching desde snapshot con `update_state` |
| 11 | `11_Creating_a_Planner_Node_withaStructured_Executor.ipynb` | Planner + ejecutor estructurado con plan JSON validado |
| 12 | `12_Building_MultiStep_Task_Chains.ipynb` | Cadena multi-paso `plan → execute → review → revise` |
| 13 | `13_Implementing_a_Supervisor_Node_and_Worker_Agents.ipynb` | Supervisor + worker agents |
| 14 | `14_Message_Passing_Across_Agent_Nodes.ipynb` | Message passing: generator → reviewer → refiner |
| 15 | `15_Debate_Agents_with_Consensus_Voting_using_LangGraph.ipynb` | Debate de agentes con votación de consenso |
| 16 | `16_Building_a_Multi-Agent_Subgraph_Workflow.ipynb` | Subgrafo multi-agente (researcher → summarizer) |

> Los notebooks `07`–`12` son interactivos (usan `interrupt()` / `input()`). Para evitar
> prompts por consola, invoca el grafo y reanúdalo con `Command(resume=...)` como muestra
> la celda "Ejemplo de uso" de cada uno.

### Patrones aplicados con datasets (17–29)

| # | Notebook | Patrón | Modelo sugerido |
|---|---|---|---|
| 17 | `17_ReAct_Deteccion_Fraude_Financiero.ipynb` | ReAct | `claude-sonnet-4-6` |
| 18 | `18_Supervisor_Customer_Support_Tickets.ipynb` | Supervisor / Hub-and-Spoke | `claude-sonnet-4-6` |
| 19 | `19_Hierarchical_MultiAgent_CustomerSupportTicketSystem.ipynb` | Multi-agente jerárquico | `claude-sonnet-4-6` |
| 20 | `20_PromptChaining_Pipeline_ConsumerComplaints.ipynb` | Prompt Chaining / Pipeline | `claude-haiku-4-5-20251001` |
| 21 | `21_parallel_mapReduce_Fanout_Fanin.ipynb` | Map-Reduce (fan-out/fan-in) con `Send()` | `claude-haiku-4-5-20251001` |
| 22 | `22_Reflection_SelfCritique.ipynb` | Reflection / Self-Critique | `claude-sonnet-4-6` |
| 23 | `23_Plan_and_Execute_Analisis_SupplyChain.ipynb` | Plan-and-Execute | `claude-sonnet-4-6` |
| 24 | `24_human_in_the_loop_LoanApproval_prediction.ipynb` | HITL (aprobación de préstamos) | `claude-sonnet-4-6` |
| 25 | `25_Agentic_RAG_Sistema_QA_Medico.ipynb` | Agentic RAG (ChromaDB) | `claude-haiku-4-5` |
| 26 | `26_CodeAgent_Analisis_Inteligente_GithubIssues.ipynb` | CodeAct Agent (**entorno aislado**) | `claude-haiku-4-5-20251001` |
| 27 | `27_CUA_ComputerUserAgent_Monitoreo_precios.ipynb` | Computer Use Agent (Playwright) | `claude-haiku-4-5` |
| 28 | `28_Memoria_GuardRails_EcommerceCustomerSupportAgent.ipynb` | Memoria + GuardRails | `claude-haiku-4-5` |
| 29 | `29_CUA_HITL_Procesamiento_Reclamos_insurances.ipynb` | CUA + HITL | `claude-sonnet-4-6` |

> *(El archivo auxiliar `temp_save_graph.ipynb` solo reconstruye/guarda el grafo del
> notebook 01; no es un patrón.)*

---

## Datos necesarios

La data vive en `data/` (salvo el dataset PaySim, que está ignorado por su tamaño:
`data/PS_*.csv`). Cada notebook trae la celda de descarga correspondiente; aquí tienes el
resumen para preparar todo desde cero.

### Credenciales para descargar datasets

- **Kaggle** (notebooks 17, 18, 20, 21, 23, 24, 25, 26, 29): coloca `~/.kaggle/kaggle.json`
  (`chmod 600`).
- **Hugging Face** (notebooks 19, 28): suele funcionar sin login; si te pide auth, exporta
  `HF_TOKEN` en el `.env`.

### Tabla de datasets por notebook

| # | Dataset | Origen / comando de descarga | Ruta destino |
|---|---|---|---|
| 17 | PaySim (~471 MB) | Kaggle `ealaxi/paysim1` (descarga manual) | CSV en la raíz; o exporta `PAYSIM_CSV=/ruta/al/PS_..._log.csv` |
| 18 | Customer Support Tickets | Kaggle `suraj520/customer-support-ticket-dataset` | `data/customer_support_tickets.csv` |
| 19 | Customer Support Tickets | HF `load_dataset("Tobi-Bueck/customer-support-tickets")` (auto) | caché de Hugging Face |
| 20 | Consumer Complaints | `kagglehub.dataset_download("shashwatwork/consume-complaints-dataset-fo-nlp")` (auto) | caché de kagglehub |
| 21 | Amazon Fine Food Reviews | Kaggle `snap/amazon-fine-food-reviews` | `data/amazon/.../reviews_Grocery_and_Gourmet_Food_5.json.gz` |
| 22 | arXiv papers | Cache local (descarga vía API de arXiv en el notebook) | `data/arxiv/arxiv_papers.csv` |
| 23 | DataCo Supply Chain | CSV público (descarga automática por URL); alt. Kaggle `shashwatwork/dataco-smart-supply-chain-for-big-data-analysis` | URL directa (sin archivo local) |
| 24 | Loan Approval Prediction | `!kaggle datasets download -d architsharma01/loan-approval-prediction-dataset --unzip -p ./data/` | `data/loan_approval_dataset.csv` |
| 25 | MedQuAD (47k Q&A médicos) | `!kaggle datasets download -d pythonafroz/medquad-medical-question-answer-for-ai-research --unzip -p ./data/medquad` | `data/medquad/medquad.csv` (+ índice `./chroma_medquad`) |
| 26 | GitHub Issues | `kagglehub` (auto) | `data/github-issues/github_issues.csv` |
| 27 | books.toscrape.com | Ninguno: scraping en vivo con Playwright | — |
| 28 | Bitext Customer Support | HF `load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")` (auto) | caché de Hugging Face |
| 29 | Health Insurance Claims | `!kaggle datasets download -d leandrenash/enhanced-health-insurance-claims-dataset --unzip -p ./data` | `data/enhanced_health_insurance_claims.csv` |

Si no tienes credenciales de Kaggle, varios notebooks incluyen un **fallback** con datos
sintéticos o un mirror público. El notebook de RAG médico (25) construye un índice local de
**ChromaDB** (`./chroma_medquad`) la primera vez que se ejecuta.

---

## Cómo ejecutar los notebooks

```bash
# Lanzar Jupyter Lab con el entorno principal
uv run jupyter lab
```

1. Abre el notebook deseado y selecciona el kernel del entorno principal
   (p. ej. *"Python (langgraph-tutorials)"* o el `.venv` del proyecto).
2. **Ejecuta las celdas de arriba abajo.** Es importante: las primeras celdas instalan
   dependencias puntuales y aplican parches de compatibilidad necesarios antes de crear el LLM.
3. Asegúrate de haber descargado la data del notebook (sección anterior) antes de las celdas
   que la cargan.

> El **notebook 26** requiere un kernel propio: ver la sección siguiente.

---

## Entorno aislado para el notebook 26

`langgraph-codeact==0.1.x` exige `langgraph<0.4`, **incompatible** con el `langgraph` 1.x del
entorno principal. Por eso el notebook 26 tiene su propio entorno en `envs/notebook26/`
(Python 3.12, stack 0.3.x):

```bash
# Crear/sincronizar el entorno aislado
uv sync --project envs/notebook26

# Registrar su kernel de Jupyter
uv run --project envs/notebook26 python -m ipykernel install --user \
    --name notebook26-codeact --display-name "Python 3.12 (notebook26 codeact)"
```

(Si `envs/notebook26/run.sh` está presente, también puedes usarlo como atajo.)

Luego abre `26_CodeAgent_*.ipynb` y selecciona el kernel **"Python 3.12 (notebook26 codeact)"**.
No instales `langgraph-codeact` en el `.venv` principal: degrada `langchain-core` y rompe el
resto de notebooks.

---

## Posibles problemas (issues) y soluciones

Estos son los errores más frecuentes al usar el repo desde cero y cómo resolverlos.

**1. `ModuleNotFoundError: No module named 'langgraph_codeact'` (notebook 26).**
Estás usando el kernel equivocado. El notebook 26 va con el entorno `envs/notebook26`
(stack 0.3.x). Selecciona el kernel *"Python 3.12 (notebook26 codeact)"*
(ver [su sección](#entorno-aislado-para-el-notebook-26)).

**2. `ImportError: cannot import name 'ContextOverflowError' from 'langchain_core.exceptions'`
(notebook 26).**
Mismo origen: estás corriendo el 26 en el `.venv` principal (stack 1.x). `langgraph-codeact`
necesita `langchain-core` 0.3.x. Usa el entorno aislado del notebook 26.

**3. `AttributeError: module 'langchain' has no attribute 'verbose'` (o `'llm_cache'`).**
Desajuste entre el paquete `langchain` y `langchain-core`: `langchain-core` lee globals
(`verbose`/`debug`/`llm_cache`) del módulo `langchain`, que en algunas versiones ya no los
expone. Los notebooks 17–29 **ya incluyen un parche de compatibilidad** que define esos
atributos y envuelve `get_verbose`/`get_llm_cache`. Solución: **ejecuta las celdas de arriba
abajo** para que el parche corra antes de instanciar `ChatAnthropic`. Para eliminar la causa
raíz, alinea versiones con `uv sync`.

**4. `your OS is not supported` / `Playwright does not support chromium on ubuntu26.04-x64`
(notebooks 27 y 29).**
Playwright no publica build de Chromium para versiones de SO muy nuevas. El notebook 27 ya
fuerza la plataforma compatible vía variable de entorno:

```python
import os
os.environ["PLAYWRIGHT_HOST_PLATFORM_OVERRIDE"] = "ubuntu24.04-x64"  # ajusta tu versión/-arm64
```

y usa `!playwright install chromium` **sin** `--with-deps`. Si al lanzar el navegador faltan
librerías del sistema, instálalas a mano (Ubuntu 24.04/26.04 usan sufijo `t64`):

```bash
sudo apt-get update && sudo apt-get install -y \
    libnss3 libnspr4 libatk1.0-0t64 libatk-bridge2.0-0t64 libcups2t64 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
    libasound2t64 libatspi2.0-0t64 libpango-1.0-0 libcairo2
```

Si falla por *sandbox* (entornos headless/contenedores), lanza con
`chromium.launch(headless=True, args=["--no-sandbox"])`.

**5. `/.../python: No module named pip` al instalar dentro de un notebook.**
Los venv creados con `uv` no incluyen `pip`. Bootstrápealo con la stdlib y reintenta:

```python
import subprocess, sys
subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "<paquete>"], check=True)
# alternativa: uv pip install --python <ruta/al/.venv/bin/python> <paquete>
```

**6. `403`/`401` o "dataset not found" al descargar data.**
Falta configurar credenciales: `~/.kaggle/kaggle.json` para Kaggle o `HF_TOKEN` para Hugging
Face (ver [Datos necesarios](#datos-necesarios)).

**7. El `.venv` principal quedó con versiones raras tras experimentar con el notebook 26.**
Restáuralo a lo fijado en `uv.lock`:

```bash
uv sync
```

**8. Los notebooks HITL (07–12, 24, 29) parecen "colgarse".**
Es esperado: el grafo se **pausa** en `interrupt()`. Reanúdalo con
`app.invoke(Command(resume=<decisión>), config=config)` usando el mismo `thread_id`; la celda
"Ejemplo de uso" de cada notebook indica el esquema exacto de la decisión.

---

## Modelos Claude disponibles

| Modelo | ID | Uso recomendado |
|---|---|---|
| Opus 4.7 | `claude-opus-4-7` | Tareas complejas con razonamiento profundo |
| Sonnet 4.6 | `claude-sonnet-4-6` | Balance velocidad / calidad (default) |
| Haiku 4.5 | `claude-haiku-4-5-20251001` | Pipelines con muchas llamadas, baja latencia |

Cambia el modelo editando `ANTHROPIC_MODEL` en `.env` o pasándolo directamente a
`ChatAnthropic(model="...")` en la celda correspondiente.

---

## Estructura del repo

```
.
├── 01_*.ipynb … 06_*.ipynb         # Fundamentos: estado, checkpoints, branching, paralelo
├── 07_*.ipynb … 16_*.ipynb         # Patrones intermedios: HITL, planner, multi-agente
│   └── 07_*/ … 16_*/               # Código fuente original (paquetes .py de referencia)
├── 17_*.ipynb … 29_*.ipynb         # Patrones aplicados sobre datasets reales
├── envs/
│   └── notebook26/                 # Entorno aislado para CodeAct (langgraph<0.4, Python 3.12)
│       ├── pyproject.toml
│       ├── uv.lock
│       ├── README.md
│       └── run.sh
├── data/                           # Datasets descargados (data/PS_*.csv ignorado por tamaño)
├── .env                            # (ignorado) ANTHROPIC_API_KEY, etc.
├── .env.example                    # Plantilla de variables de entorno
├── pyproject.toml                  # Entorno principal (Python 3.13, langgraph 1.x)
└── uv.lock
```

---

## Licencia

Ver `LICENSE`.
