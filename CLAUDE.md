# Working in this repository

## The paper is published: version every change

Paper v1 was sent out on 21 October 2026. It is frozen, and so is everything that is part of it:

- `study/paper/paper.md`
- `study/paper/can-creativity-be-reverse-engineered.pdf`
- the `study/paper/fig*.png` figures
- its Claude Doc (https://claude.ai/artifact/U2ifBBtEGYfSdia2FjeaZy)
- the git tag `paper-v1`

Do not edit any of them.

Any change to the paper goes into a new version folder, `study/paper/v2/`. Start it as a copy of v1; it has its own PDF and its own Claude Doc titled "(v2 draft)". Keep making changes in that same draft until Mike says it is published. Then tag it, mark it published, and start the next version for later changes. The full rules and commands are in [`study/paper/README.md`](study/paper/README.md).

### The v2 draft exists

A v2 draft was started on 7 November 2026. It is not published.

- Text: `study/paper/v2/paper.md`, with `CHANGES.md`, the `fig_*.png` figures and the PDF beside it.
- Claude Doc (v2 draft): https://claude.ai/artifact/KTmxrnErgejqUBU6apgzNy. Keep it in step with `v2/paper.md`: every change goes into both.
- Code and plan: `study/v2/` (`PLAN.md` is frozen; later changes are logged in its Section 18). Results: `study/results/v2/`.
- Its byline is "Mike Taylor and Claude (Anthropic)". Do not put a model version in it.
- Rebuild the tables and figures with `bash v2/make_paper_assets.sh` from `study/`, then the PDF with `paper/build_pdf.py paper/v2`.

## Layout

- `study/`: the Batman paper and everything that produced it. Run its commands from `study/` with `../.venv/bin/python`.
- `study/v2/`: the code behind the v2 draft (the GPU pipeline and the analyses). Its large arrays (`v2/acts/`) and corpora (`v2/cache/`) are not committed.
- `images/`: the exploratory image side experiment. It is not part of the paper.
- `preregistration/`: the planned next experiments. They are not run; read `preregistration/README.md` before resuming.

## Other standing instructions

- Do not cite or name Mike's book *Marketing Memetics* in the paper, the companion page or the READMEs. The links Mike shared (his talk deck, his Notion essays, his r/memetics post) are public and can be linked.
- The companion page (https://claude.ai/artifact/EY1zb223bqB591NSN34hes) is shared publicly with anyone who has the link. Republishing it changes what readers see straight away.
- Every number in the paper and on the page must come from a committed output in `study/results/`.
