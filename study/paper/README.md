# The paper: versions

*Can Creativity Be Reverse-Engineered? Language-model neurons as units of cultural inheritance in 80 years of Batman*, by Mike Taylor and Claude Opus 5.5.

| Version | Status | Where | Git tag |
| --- | --- | --- | --- |
| v1 | **Published** 21 October 2026 (sent out) | This folder: [`paper.md`](paper.md), [`can-creativity-be-reverse-engineered.pdf`](can-creativity-be-reverse-engineered.pdf) and the `fig*.png` figures. Claude Doc: https://claude.ai/artifact/U2ifBBtEGYfSdia2FjeaZy | `paper-v1` |

## Rules

The paper has been sent out, so published versions are frozen. Every change goes into a new version.

1. **A published version never changes.** That covers its `paper.md`, its PDF, its figures and its Claude Doc. The links people were sent keep pointing at exactly what they read. The git tag records each published version (`git checkout paper-v1`).
2. **To change anything, start the next version.** Create `v2/` here and copy in v1's text and figures, then edit only `v2/`:
   ```bash
   # from study/
   mkdir paper/v2 && cp paper/paper.md paper/fig*.png paper/v2/
   ```
   - Build its PDF with `../.venv/bin/python paper/build_pdf.py paper/v2`. That writes `paper/v2/can-creativity-be-reverse-engineered-v2.pdf`.
   - Regenerate a figure into it with `PAPER_DIR=paper/v2 ../.venv/bin/python paper/figs/canon.py` (or `lineage.py`, `culture.py`).
   - Keep a `v2/CHANGES.md` listing what changed from v1, and why.
   - Make a new Claude Doc for the draft by copying v1's Doc, title it with "(v2 draft)", and keep it in sync with `v2/paper.md`. Never edit v1's Doc.
3. **Keep changing the same draft until Mike says it is published.** All changes in the meantime go into that draft; don't start v3.
4. **When Mike says it is published:** mark it Published in the table above with the date, tag the commit `paper-v2`, and remove "draft" from its Doc title. The next change after that starts `v3/`.
5. **Code and results can move on, but a new version must say what changed.** If the study's code or results change in a way that affects a published number, the new version reports the change in its `CHANGES.md` and its text. Nothing silently contradicts a published version.
