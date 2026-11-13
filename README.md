# Smart Question Router

Meet Suvra's simplest "Neuroscience Oracle Bot" that can make tough decision!

```text
Question -> Jev classification (category + confidence)
         -> Python triage rules
         -> One LLM, using a Lateral Hypothalamus or Hippocampus system prompt -> Answer
         -> Or: clarification, without calling the LLM
```

## Run locally

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```powershell
uv sync --locked
Copy-Item .env.example .env
# Edit .env and enter both API keys.
uv run --locked streamlit run app.py
```

Get a TypeSafe key from the [TypeSafe console](https://console.typesafe.ai/) and an
OpenAI key from the [OpenAI platform](https://platform.openai.com/api-keys).
Each provider needs API access and available credits. The app loads `.env` from
the project directory; existing environment variables take precedence. Never
commit `.env`. Questions are sent to Jev; routed questions are also sent to OpenAI.
Both APIs also receive the recent conversation context described below.

## Chatting

Type a message in the chat box and press Enter. Each user message shows its
detected category, Jev confidence, and selected route. Answers and clarification
requests appear as assistant messages.

You can ask follow-ups (for example, "Can you give an example?") or switch from
the lateral hypothalamus to the hippocampus. Every new message is classified again. Both Jev and the
answering model receive the current message plus the preceding **three successful
exchanges** (up to six messages), in chronological order. Clarification exchanges
count as successful: reply with context to clarify an earlier question.

Older messages remain visible but are no longer sent to the APIs. API failures
stay visible alongside their user messages; both messages of a failed exchange
are excluded from future context. Classification metadata is kept out of model
context. Responses stay concise, using the prompt chosen for the current message.

Click **New chat** to clear the conversation and its context. Messages live only
in the current Streamlit session, with no database or disk persistence. A new
browser session starts a separate conversation.

Configuration:

| Variable | Purpose | Default |
| --- | --- | --- |
| `TYPESAFE_API_KEY` | Jev classification credentials | Required |
| `OPENAI_API_KEY` | Answer generation credentials | Required for domain answers |
| `TYPESAFE_DEFAULT_MODEL` | Jev model | `jev-1.14.0` |
| `OPENAI_MODEL` | Single answer model for both domains | `gpt-4.1-mini` |

## Architecture

| File | Responsibility |
| --- | --- |
| `classifier.py` | Calls Jev through `typesafe-sdk`; returns `Classification(category, confidence)` |
| `triage.py` | Applies deterministic rules; returns a route and optional clarification |
| `llm.py` | Calls OpenAI Responses with one of two system prompts |
| `app.py` | Streamlit input, orchestration and result display |
| `conversation.py` | Shared message format and three-exchange context limit |
| `evaluate.py` | Measures live classification and routing accuracy |
| `tests/` | Offline unit/integration tests and 28 labeled evaluation questions |

The two “assistants” are names for prompts sent to the same model. There are no
agent objects, tools, autonomous loops or orchestration frameworks.

## Classification versus triage

Jev receives a three-option Choice with descriptions of `LATERAL_HYPOTHALAMUS`, `HIPPOCAMPUS`
and `UNKNOWN`. The domains cover the lateral hypothalamus (LH/LHA) and the
hippocampus/hippocampal formation. A focused question can mention the other region
as background without becoming mixed. Comparisons or interactions genuinely
requiring both regions, insufficient context, or neither region should be `UNKNOWN`.
Broad words such as memory or feeding alone do not establish which region is meant.
These are classification instructions, not guarantees about model accuracy.

Triage has a fixed default confidence threshold of **0.70**, defined in `triage.py`:

| Jev result | Route |
| --- | --- |
| `LATERAL_HYPOTHALAMUS`, confidence >= 0.70 | Lateral Hypothalamus Assistant |
| `HIPPOCAMPUS`, confidence >= 0.70 | Hippocampus Assistant |
| Either known category, confidence < 0.70 | Ask for clarification |
| `UNKNOWN`, any confidence | Ask for clarification |

For example, `HIPPOCAMPUS` at 0.60 remains the detected category, but triage asks for
clarification. `UNKNOWN` at 0.99 also asks for clarification: Jev can be certain a
question does not fit. A failed API request displays an error, not an `UNKNOWN`
classification. Clarify by sending a follow-up message with more context.

Jev's confidence is derived from its probability distribution; it is not the
top-class probability or a measured accuracy percentage. We preserve the returned
score. The threshold is an initial teaching choice and needs validation on real
questions. Routing is deterministic given a classification; model outputs can vary.

## Test and evaluate

```powershell
uv run --locked pytest
```

Tests require no API keys and make no external calls. They exercise the installed
SDKs using fake HTTP responses, confidence boundaries, error handling, the
Streamlit chat flow, conversation limits and evaluation scoring. They do **not**
measure Jev accuracy. The 28-question evaluation still classifies each question
independently, without chat history.

The 28 questions in `tests/questions.json` include 11 lateral hypothalamus, 11 hippocampus,
3 ambiguous, 2 mixed-domain and 1 unrelated case. Expected labels reflect this
project's conservative policy: ambiguous and genuinely mixed requests need clarification.

Each domain includes four additional questions marked `"difficulty": "advanced"`
(IDs 21–24 for the lateral hypothalamus and 25–28 for the hippocampus). Each combines
omics, gene variants, circuit projections and DREADD experiment interpretation.
They challenge causal inference, cell-state versus cell-composition effects,
projection collaterals, ligand effects and rescue versus compensation. Proposed
variant-to-circuit links are research scenarios, not claims that those experiments
or associations have been established. Unspecified candidate variants are explicitly
hypothetical. These remain classification/routing questions; the evaluation does
not grade the scientific quality of generated answers.

Scientific background for these scenarios includes
[projection-resolved lateral hypothalamic transcriptomics](https://pubmed.ncbi.nlm.nih.gov/34624220/),
[BDNF Val66Met and ventral CA1 projections](https://pubmed.ncbi.nlm.nih.gov/29909994/),
[APOE4 effects in human brain cell models](https://pubmed.ncbi.nlm.nih.gov/29861287/),
[TREM2 R47H and microglial responses](https://pubmed.ncbi.nlm.nih.gov/32959884/),
[FTO rs1421085 and adipocyte mechanisms](https://pubmed.ncbi.nlm.nih.gov/26287746/), and
[DREADD ligand pharmacology](https://pubmed.ncbi.nlm.nih.gov/30872749/).

To measure the real classifier after setting `TYPESAFE_API_KEY`:

```powershell
uv run --locked python evaluate.py
```

This makes **28 billable Jev calls** and no OpenAI calls. It prints per-question
results and saves `evaluation-results.json` with predictions, confidences, expected
labels/routes, model, timestamp, threshold and errors. Classification accuracy is
the fraction with the correct label; routing accuracy is the fraction with the
correct final route. Both use all 28 questions as the denominator; API errors
count as failures and are also reported separately. The command exits nonzero on
API errors. Inspect failures before adjusting the threshold, and use a separate,
larger held-out dataset for broader claims. Answer quality is outside this evaluation.

No live accuracy result is bundled: it requires your TypeSafe credentials.

## API references

- [TypeSafe Python SDK](https://docs.typesafe.ai/sdk/python)
- [Jev Choice and confidence](https://docs.typesafe.ai/confidence)
- [Jev model versions](https://docs.typesafe.ai/models)
- [OpenAI Responses API](https://developers.openai.com/api/reference/python/resources/responses/methods/create)
- [Streamlit app testing](https://docs.streamlit.io/develop/api-reference/app-testing)
