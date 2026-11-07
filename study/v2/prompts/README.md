# Prompts for the AI agents

These are the instructions given to the AI agents of PLAN.md, Sections 7, 10 and 11, word for word. Each agent is a fresh one with no memory of the others, and works from its prompt alone. `{...}` marks what is filled in per batch. The prompts were committed with the plan, before any study unit was looked at.

| File | Role | Sees |
| --- | --- | --- |
| `explainer.md` | Writes a unit's title, explanation and key words | The unit's strongest examples |
| `keyword_analyst.md` | Writes the search terms of an analyst who expects evasion | Title and explanation only |
| `tester.md` | Writes the test texts A, B, C, D and M | Title, explanation and the explainer's key words only |
| `kind_judge.md` | Says what kind of thing the unit detects | Title and explanation only |
| `steering_judge.md` | Picks which concept was pushed into a continuation | Continuations and five candidate titles |
| `identify_judge.md` | Says whether a reply names a given work | A reply and the true title |
