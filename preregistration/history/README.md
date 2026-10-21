# history/

**Note on citations.** Some of these files cite a book by Mike Taylor as the source of the orphan-vigilante family (the memes Batman shares with the other "billionaire vigilante orphans", which form Tier A) and of its tiers. From prereg-v3 (2026-10-21) the book is no longer cited anywhere in this pre-registration: its ideas are presented without the citation, and the names that referred to it were renamed: in `culture_pool.json` the role `book_family` became `orphan_family` and the fields `book_tier` and `book_entity` became `family_tier` and `family_entity`; in `documented_relatives.json` and `sources/relatives_documented_relatives.json` the relations `book_tier1` and `book_tier2` became `family_tier1` and `family_tier2`, the strength `book` became `family`, and one clause citing the book was deleted from an evidence note; the tier A label in `memes_for_tracing.json` no longer names a source. The files in this folder keep their original text.

Earlier versions of the design, kept verbatim as a record (`PREREGISTRATION.md` section 14):

- `PREREGISTRATION_draft_v1.md`, `memes_for_tracing_draft_v1.json`, `planted_design_draft_v1.json`: the v1 drafts (2026-10-20), before the AI design review.
- `power_sim_v2.py`, `power_v2.json`: the prereg-v2 power simulation (five primary claims).
- `power_sim_v3.py`, `power_v3.json`: the prereg-v3 power simulation (H1 to H3 under Holm over three, S-coded and the overlap test), with one rival and no rival signal: single-rival upper bounds for H1 and H2.
- `power_sim_v4.py`, `power_v4.json`: the power simulation after the fifth review (`PREREGISTRATION.md` section 14.5): H1 and H2 with their actual intersection-union tests against correlated rivals that have signal, H2 at its non-inferiority margin, and S-coded and the overlap test at the prominent family's Benjamini-Hochberg thresholds. Section 6 quotes it (H3 still from v3).
