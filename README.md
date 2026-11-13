## Smart Question Router

Meet Suvra's simplest "Neuroscience Oracle Bot" that can (?) make tough decision! Well, not yet!

I was wandering about how to incorporate better decision making with "AI Systems Design" while dealing with tough scientific questions, and came across this recently lauched JEV from Typesafe. This repo is a small playground experiment of my never-ending learning journey. I have tried following:

`Question → Jev classification → Python triage → Domain-specific LLM answer`

Jev returns a category and confidence; deterministic rules choose a route;
one OpenAI model answers using the selected region's system prompt.

## Run locally

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```powershell
uv sync --locked
Copy-Item .env.example .env  # Skip if .env already exists
```

Set `TYPESAFE_API_KEY` and `OPENAI_API_KEY` in `.env`.
Get keys from [TypeSafe](https://console.typesafe.ai/) and
[OpenAI](https://platform.openai.com/api-keys). Keep `.env` private.

```powershell
uv run --locked streamlit run app.py
```

Defaults: `jev-1.14.0` and `gpt-4.1-mini`; configurable in `.env`.

## Behavior

- Displays category, confidence, route, and answer.
- Routes known categories at confidence **≥ 0.70**; otherwise asks for clarification.
- Mixed-region and out-of-scope questions also request clarification.
- Uses the last **three successful exchanges** for follow-ups; excludes API failures.
- Keeps chat in the current session. **New chat** clears it.

## Test and evaluate

```powershell
uv run --locked pytest              # Offline; no API keys needed
uv run --locked python evaluate.py  # 28 billable Jev calls
```

The **28-question dataset** includes four advanced questions per region on omics,
gene variants, projections, and DREADDs. Hypothetical mechanisms are labeled.

Live evaluation saves classification and routing accuracy to
`evaluation-results.json`. API errors count as failures; answer quality is not
graded. Jev confidence is not measured accuracy.

### Evaluation results

```text
Completed: 28/28; errors: 0
Classification accuracy: 100.0%
Routing accuracy: 96.4%
Denominator: all 28 questions; API errors count as failures.
```

Full details: [evaluation-results.json](evaluation-results.json).

## Personal opinion

| Positive | Confused | Negative |
| --- | --- | --- |
| ![A straightforward hippocampus question routes successfully.](docs/images/jev-positive.png) | ![An explicit scientific question still triggers clarification.](docs/images/jev-negative.png) | ![An unrelated food request receives a neuroscience clarification.](docs/images/jev-confused.png) |
| A clear question routes successfully. | A detailed scientific question gets stalled. | A food request gets a neuroscience follow-up 😜 |

JEV chose the expected lateral hypothalamus category, but its confidence was 0.50 in the saved evaluation. My 0.70 rule therefore stopped the request. The label was correct; the application still asked the user to clarify. I found that frustrating because the intended topic was explicit. But this single example does not show why confidence dropped. It gives me a concrete case to investigate: should I improve the classification instructions, change the fallback, or test a different threshold? Lowering the threshold alone could let more incorrect routes through elsewhere. 

I also need to read & interpret confidence carefully. TypeSafe describes it as a summary of how concentrated the model's probabilities are across the available choices. A confidence score is not a measured success rate for my scientific questions. That needs its own evaluation.

A request for a spicy food was classified as unknown, as I would expect from this narrow tool. My app then asked whether the user meant the lateral hypothalamus or hippocampus. Dinner had become a neuroscience consultation. That reply came from my fixed fallback message. I had grouped unclear, mixed-topic, and unrelated requests into one bucket, then treated them all alike. A better response would explain the tool's scope when someone asks about restaurants and reserve clarification for questions it could actually help with.


**Next-Steps:** My instinct now is to improve the prompts and fallback behavior before adding another agent. I want to try more varied scientific questions, repeat the evaluation, and compare models on the same task. I also want expert review of the generated answers, because routing and scientific reasoning deserve separate checks.For now, I see JEV as a promising component for a clearly defined decision. This experiment made me pay closer attention to the rules around it. I'll keep sharing my hands-on experience as I push this model toward
scalable production systems and tougher scientific decisions. Can better prompts close the gap, or do we need a better model for scientific decision-making? How will that differ when multiple scientific variables comes into play? Stay tuned, will be updating soon in my next cross model comparision in multi-agentic system design repo :)  
