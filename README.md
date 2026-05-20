# Langgraph Tutorials

Colección práctica de **scripts** y **notebooks** que ilustran patrones de agentes con
[LangGraph](https://langchain-ai.github.io/langgraph/) usando **Claude (Anthropic)** como LLM.
Los ejemplos van desde el manejo básico de estado (scripts `01`–`06`) hasta arquitecturas
multi-agente, RAG agéntico, Computer Use Agents y HITL en notebooks `17`–`30`.

> Toda la integración LLM está unificada en **Anthropic Claude**. La API key se carga vía
> `python-dotenv` desde `.env`.

---

## Tabla de contenidos

- [Setup inicial](#setup-inicial)
- [Configuración del `.env`](#configuración-del-env)
- [Scripts (01–16)](#scripts-01-16)
- [Notebooks (17–30)](#notebooks-17-30)
- [Entorno aislado para el notebook 26](#entorno-aislado-para-el-notebook-26)
- [Cómo ejecutar](#cómo-ejecutar)
- [Datasets](#datasets)
- [Estructura del repo](#estructura-del-repo)

---

## Setup inicial

Requisitos: **Python ≥ 3.13** y [uv](https://docs.astral.sh/uv/) como gestor de paquetes.

```bash
# 1. Clonar
git clone <repo-url> && cd Langgraph_tutorials

# 2. Sincronizar venv principal (Python 3.13)
uv sync

# 3. Copiar plantilla de variables y editarla
cp .env.example .env
$EDITOR .env   # poner tu ANTHROPIC_API_KEY
```

El comando `uv sync` instala todas las dependencias declaradas en `pyproject.toml`
(LangChain, LangGraph, pandas, matplotlib, ChromaDB, sentence-transformers, etc.).

## Configuración del `.env`

```env
# Anthropic Claude API
ANTHROPIC_API_KEY=sk-ant-...

# Modelo por defecto (configurable por script)
# Opciones: claude-opus-4-7, claude-sonnet-4-6, claude-haiku-4-5-20251001
ANTHROPIC_MODEL=claude-sonnet-4-6
```

`python-dotenv` lo carga automáticamente. Algunos notebooks usan también `KAGGLE_USERNAME` /
`KAGGLE_KEY` o `HUGGINGFACEHUB_API_TOKEN` para descargar datasets — opcionales y
documentados en cada notebook.

---

## Scripts (01–16)

Scripts `.py` que cubren los fundamentos de LangGraph. Cada uno se ejecuta de forma
independiente desde la raíz del repo.

| # | Archivo / Carpeta | Patrón / Tema | Cómo correr |
|---|---|---|---|
| 01 | `01_langgraph_state_typed.py` | StateGraph con `TypedDict` tipado | `uv run python 01_langgraph_state_typed.py` |
| 02 | `02_managing_graph_state.py` | Reducers (`Annotated[Sequence, operator.add]`) | `uv run python 02_managing_graph_state.py` |
| 03 | `03_persistent_checkpoint.py` | Checkpoint SQLite persistente | `uv run python 03_persistent_checkpoint.py` |
| 04 | `04_multibranch_execution.py` | Multi-branch con `Send()` | `uv run python 04_multibranch_execution.py` |
| 05 | `05_conditional_router.py` | Routing condicional con `add_conditional_edges` | `uv run python 05_conditional_router.py` |
| 06 | `06_parallel_execution_and_subgraph.py` | Ejecución paralela + subgraph | `uv run python 06_parallel_execution_and_subgraph.py` |
| 07 | `07_implementing_human_in_the_loop/` | HITL básico — aprobación de documentación técnica | `uv run python 07_implementing_human_in_the_loop/main.py` |
| 08 | `08_Designing_a_Multi-Stage_Approval_Workflow/` | HITL multi-etapa (code review → testing → manager → UX) | `uv run python 08_Designing_a_Multi-Stage_Approval_Workflow/main.py` |
| 09 | `09_Using_Execution_Log_and_Resuming_from_Checkpoints/` | Replay desde un `checkpoint_id` | `uv run python 09_Using_Execution_Log_and_Resuming_from_Checkpoints/main.py` |
| 10 | `10_Performing_Branch_Analysis_with_Snapshots/` | Branching desde snapshot con `update_state` | `uv run python 10_Performing_Branch_Analysis_with_Snapshots/main.py` |
| 11 | `11_Creating_a_Planner_Node_withaStructured_Executor/` | Planner → executor estructurado con plan JSON validado | `uv run python 11_Creating_a_Planner_Node_withaStructured_Executor/main.py` |
| 12 | `12_Building_MultiStep_Task_Chains/` | Cadena multi-paso plan→execute→review→revise | `uv run python 12_Building_MultiStep_Task_Chains/main.py` |
| 13 | `13_Implementing_a_Supervisor_Node_and_Worker_Agents/` | Supervisor + worker agents | `uv run python 13_Implementing_a_Supervisor_Node_and_Worker_Agents/main.py` |
| 14 | `14_Message_Passing_Across_Agent_Nodes/` | Message passing: generator → reviewer → refiner | `uv run python 14_Message_Passing_Across_Agent_Nodes/main.py` |
| 15 | `15_Debate_Agents_with_Consensus_Voting_using_LangGraph/` | Debate de agentes con votación de consenso | `uv run python 15_Debate_Agents_with_Consensus_Voting_using_LangGraph/main.py` |
| 16 | `16_Building_a_Multi-Agent_Subgraph_Workflow/` | Parent graph + subgraph multi-agente | `uv run python 16_Building_a_Multi-Agent_Subgraph_Workflow/main.py` |

**Notas**

- Los scripts `07–12` son interactivos: usan `interrupt()` y piden input por terminal.
- `13–16` usan un módulo compartido `llm_provider.py` que centraliza la instanciación
  de `ChatAnthropic` con `ANTHROPIC_API_KEY` y `ANTHROPIC_MODEL`.

---

## Notebooks (17–30)

Cada notebook implementa un **patrón de agente** sobre un dataset real (Kaggle / HuggingFace
/ mirror público). Todos usan Claude vía `langchain-anthropic`.

| # | Notebook | Patrón | Dataset | Modelo sugerido |
|---|---|---|---|---|
| 17 | `17_ReAct_Deteccion_Fraude_Financiero.ipynb` | **ReAct** | PaySim (Kaggle, ~470 MB) | `claude-sonnet-4-6` |
| 18 | `18_Supervisor_Customer_Support_Tickets.ipynb` | **Supervisor / Hub-and-Spoke** | suraj520/customer-support-ticket-dataset (Kaggle) | `claude-sonnet-4-6` |
| 19 | `19_Hierarchical_MultiAgent_CustomerSupportTicketSystem.ipynb` | **Hierarchical Multi-Agent** | Tobi-Bueck/customer-support-tickets (HF) | `claude-sonnet-4-6` |
| 20 | `20_PromptChaining_Pipeline_ConsumerComplaints.ipynb` | **Prompt Chaining / Pipeline** | CFPB Consumer Complaints (Kaggle) | `claude-haiku-4-5-20251001` |
| 21 | `21_parallel_mapReduce_Fanout_Fanin.ipynb` | **Parallel / Map-Reduce (Fan-out/Fan-in)** | Amazon Fine Food Reviews (Kaggle) | `claude-haiku-4-5-20251001` |
| 22 | `22_Reflection_SelfCritique.ipynb` | **Reflection / Self-Critique** | Learning Agency Lab — ASAP 2.0 (Kaggle) | `claude-sonnet-4-6` |
| 23 | `23_Plan_and_Execute_Analisis_SupplyChain.ipynb` | **Plan-and-Execute** | DataCo Supply Chain (mirror público) | `claude-sonnet-4-6` |
| 24 | `24_human_in_the_loop_LoanApproval_prediction.ipynb` | **Human-in-the-Loop (HITL)** | Loan Approval Prediction (Kaggle) | `claude-sonnet-4-6` |
| 25 | `25_Agentic_RAG_Sistema_QA_Medico.ipynb` | **Agentic RAG** | MedQuAD — 47k Q&A médicos (Kaggle) | `claude-haiku-4-5` |
| 26 | `26_CodeAgent_Analisis_Inteligente_GithubIssues.ipynb` | **CodeAct Agent** | GitHub Issues (Kaggle) | `claude-haiku-4-5-20251001` |
| 27 | `27_CUA_ComputerUserAgent_Monitoreo_precios.ipynb` | **CUA — Computer Use Agent** | books.toscrape.com (web scraping legal) | `claude-haiku-4-5` |
| 28 | `28_Memoria_GuardRails_EcommerceCustomerSupportAgent.ipynb` | **Memoria + GuardRails** | Bitext Customer Support (HF) | `claude-haiku-4-5` |
| 29 | `29_CUA_HITL_Procesamiento_Reclamos_insurances.ipynb` | **CUA + HITL** | leandrenash/enhanced-health-insurance-claims-dataset (Kaggle) | `claude-sonnet-4-5` |
| 30 | `30_LanggraphFinance.ipynb` | **Finance Agent** (RAG + tools + SQL) | datos sintéticos / SQLAlchemy local | `claude-haiku-4-5-20251001` |

> Todos los notebooks ya tienen `from dotenv import load_dotenv; load_dotenv()`
> y `ChatAnthropic(...)`. La celda `!pip install ...` puede omitirse: las dependencias ya
> están instaladas vía `uv sync`.

### Detalles de arquitectura por patrón

- **ReAct (17)**: loop `agent → call_model → tools` con `llm.bind_tools()`. El modelo
  razona y decide qué tool ejecutar (`stats`, `search_by_amount`, `flag_suspicious`, etc.).
- **Supervisor (18)**: orquestador LLM que enruta a workers especializados (billing /
  technical / refund). Implementado con `langgraph-supervisor`.
- **Hierarchical (19)**: root orchestrator → sub-orquestadores de dominio → hojas
  especializadas → quality checker. 3 niveles de subgraphs.
- **Prompt Chaining (20)**: 5 nodos LLM en serie determinística:
  `clean → classify → sentiment → summarize → recommend`.
- **Map-Reduce (21)**: `Send()` para fan-out por categoría, reducer `operator.add` para
  fan-in. Reporte ejecutivo agregado.
- **Reflection (22)**: ciclo `generate → judge → refine` hasta `score ≥ 0.8`.
- **Plan-and-Execute (23)**: planner LLM genera plan JSON, executor lo ejecuta paso a
  paso, re-planner reevalúa.
- **HITL (24)**: `interrupt()` en `analyze_risk_node` para revisión humana en zona gris.
  Incluye UI interactiva con `ipywidgets`.
- **Agentic RAG (25)**: ChromaDB local con embeddings HF + grading de relevancia +
  reformulación de query + verificación de alucinaciones.
- **CodeAct (26)**: el LLM genera **código Python ejecutable** en vez de tool calls JSON.
  Usa `langgraph-codeact`. **Requiere entorno aislado** (ver sección abajo).
- **CUA (27)**: loop observe → plan → act sobre Playwright. Tools: `goto`, `click`,
  `read_text`, `screenshot`. Caso: monitoreo de precios.
- **Memoria + GuardRails (28)**: `MessagesState` (short-term) + `InMemoryStore`
  (long-term) + guardrail PII regex + LLM judge de dominio.
- **CUA + HITL (29)**: extiende CUA con `interrupt()` antes de cada acción irreversible
  (aprobar/rechazar reclamación). Apto para producción.
- **Finance (30)**: agente con SQLAlchemy local + ChromaDB para RAG sobre reportes
  financieros sintéticos.

---

## Entorno aislado para el notebook 26

`langgraph-codeact==0.1.x` requiere `langgraph<0.4`, incompatible con el `langgraph>=1.1`
del resto del proyecto. Por eso vive en `envs/notebook26/` con su propio `pyproject.toml`,
`.venv` (Python 3.12) y kernel de Jupyter.

```bash
# Sincronizar y lanzar Jupyter Lab apuntando al notebook 26
./envs/notebook26/run.sh

# O ejecutarlo headless (CI / smoke test)
./envs/notebook26/run.sh --execute

# Solo activar la shell del venv
./envs/notebook26/run.sh --shell
```

El kernel se llama **"Python 3.12 (notebook26 codeact)"** y queda registrado en
`~/.local/share/jupyter/kernels/notebook26-codeact`. Ver `envs/notebook26/README.md`
para más detalles.

---

## Cómo ejecutar

### Scripts (01–16)

```bash
uv run python 01_langgraph_state_typed.py
uv run python 07_implementing_human_in_the_loop/main.py
# ...etc.
```

`uv run` activa el venv automáticamente. Si prefieres activarlo manualmente:

```bash
source .venv/bin/activate
python 01_langgraph_state_typed.py
```

### Notebooks (17–30, excepto 26)

```bash
# Lanzar Jupyter Lab (o tu IDE favorito apuntando al kernel del venv)
uv run jupyter lab
```

Selecciona el kernel del venv principal y ejecuta celda a celda. Recuerda que la celda
`!pip install ...` ya es redundante.

### Notebook 26

Usa el script dedicado (sección anterior):

```bash
./envs/notebook26/run.sh
```

---

## Datasets

La carpeta `data/` está en `.gitignore` — cada notebook documenta cómo descargar su
dataset. Patrones más comunes:

```python
# Kaggle (requiere ~/.kaggle/kaggle.json)
import kagglehub
path = kagglehub.dataset_download("ealaxi/paysim1")

# HuggingFace
from datasets import load_dataset
ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

# Mirror público directo
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/.../dataset.csv")
```

Si no tienes credenciales de Kaggle, los notebooks suelen incluir un **fallback**
con datos sintéticos o un mirror alternativo.

---

## Estructura del repo

```
.
├── 01_*.py … 06_*.py              # Fundamentos: state, checkpoints, branching, paralelo
├── 07_*/  … 16_*/                 # Patrones intermedios: HITL, planner, multi-agente
├── 17_*.ipynb … 30_*.ipynb        # Notebooks por patrón (ReAct, RAG, CUA, …)
├── envs/
│   └── notebook26/                # Entorno aislado para CodeAct (langgraph<0.4)
│       ├── pyproject.toml
│       ├── README.md
│       └── run.sh
├── data/                          # (ignorado) datasets descargados
├── .env                           # (ignorado) ANTHROPIC_API_KEY
├── .env.example                   # plantilla
├── pyproject.toml                 # venv principal (Python 3.13, langgraph 1.x)
└── uv.lock
```

---

## Modelos Claude disponibles

| Modelo | ID | Uso recomendado |
|---|---|---|
| Opus 4.7 | `claude-opus-4-7` | Tareas complejas que requieren razonamiento profundo |
| Sonnet 4.6 | `claude-sonnet-4-6` | Balance velocidad / calidad (default) |
| Haiku 4.5 | `claude-haiku-4-5-20251001` | Pipelines con múltiples llamadas, baja latencia |

Cambia el modelo modificando `ANTHROPIC_MODEL` en `.env` o pasándolo directamente a
`ChatAnthropic(model="...")` en el código.

---

## Licencia

Ver `LICENSE`.
