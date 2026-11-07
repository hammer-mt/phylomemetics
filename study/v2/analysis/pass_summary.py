"""What was run, for each model: checkpoint, precision, corpus sizes in its own tokens, where the scales come from.

    cd study && ../.venv/bin/python v2/analysis/pass_summary.py

Reads the small metadata files fetched with the passes (v2/acts/<tag>/meta.json, sup_meta.json, study_meta.json,
bg_meta.json, fr_meta.json, scale_summary.json) and writes results/v2/passes.json, so that the paper's numbers about
the passes come from a committed file. The checkpoint revisions were read on the GPU machines from the model cache
(hub/models--*/refs/main) after the passes.
"""
import json

import common as C

import experiments as X

REVISIONS = {"qwen3-32b": ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137"),
             "qwen3-8b": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
             "llama3.1-8b": ("unsloth/Meta-Llama-3.1-8B-Instruct", "a2856192dd7c25b842431f39c179a6c2c2f627d1"),
             "llama3.3-70b": ("unsloth/Llama-3.3-70B-Instruct", "99cd0d2c829e92a67c844f9144c2509632e5c87f")}


def load(tag, name):
    p = C.ACTS / tag / name
    return json.load(open(p)) if p.exists() else None


def scale_block(tag):
    """Which corpus set each unit's scale, and what the study texts' part in the scale does to base rates (PLAN.md Section 18, change 21)."""
    src, capn = C.scale_source(tag), C.CAP * C.n_plots(tag) + 1e-9
    cr = C.base_count(tag, which="ref")
    st = C.Setting(tag, icon=False)
    d = st.distinctive()
    dist, canon = st.units[d], st.units[d & (st.P["orig"][X.BAT].mean(0) >= 0.8)]
    return {"units_scaled_by": {name: {"all": int((src == k).sum()), "present_in_some_work": int((src[st.units] == k).sum()),
                                       "distinctive": int((src[dist] == k).sum()), "canon_all_32": int((src[canon] == k).sum())}
                                for k, name in enumerate(("reference corpus", "supplement", "study texts"))},
            "distinctive_units": int(len(dist)), "canon_units_all_32": int(len(canon)),
            "distinctive_on_the_main_scale_but_over_the_cap_on_the_reference_only_scale": int((cr[dist] > capn).sum()),
            "canon_units_over_the_cap_on_the_reference_only_scale": int((cr[canon] > capn).sum())}


def main():
    out = {}
    for tag, (repo, rev) in REVISIONS.items():
        ref, sup, study, bg, fr, sc = (load(tag, n) for n in ("meta.json", "sup_meta.json", "study_meta.json", "bg_meta.json", "fr_meta.json", "scale_summary.json"))
        if ref is None:
            continue
        assert ref["repo"] == repo, (ref["repo"], repo)
        row = {"repository": repo, "revision": rev, "layers": ref["n_layers"], "width": ref["width"], "neurons": ref["n_layers"] * ref["width"],
               "units": 2 * ref["n_layers"] * ref["width"], "precision": ref["dtype"].replace("torch.", ""),
               "reference": {"documents": ref["n_docs"], "tokens": ref["n_tokens"], "max_tokens_per_document": ref["max_tokens"],
                             "tokens_per_batch": ref["batch_tokens"], "examples_kept_per_unit": ref["k"], "seconds": ref["seconds"]}}
        if sup:
            row["supplement"] = {"documents": sup["n_docs"], "tokens": sup["n_tokens"], "max_tokens_per_document": sup["max_tokens"], "seconds": sup["seconds"]}
        if study:
            row["study"] = {"works": study["n_works"], "seconds": study["seconds"], "software": study.get("provenance")}
        if bg:
            row["background"] = {"plots": bg["n_plots"], "tokens": bg["n_tokens"], "one_plot_at_a_time": bg["one_plot_at_a_time"], "seconds": bg["seconds"]}
        if fr:
            row["franchises"] = {"films": fr["n_plots"], "tokens": fr["n_tokens"]}
        if sc:
            row["scale"] = {k: v for k, v in sc.items() if not k.endswith("_at_or_above_30pct")}
            row["share_of_unit_by_work_cells_present"] = {k.replace("_share_of_cells_at_or_above_30pct", ""): v for k, v in sc.items() if k.endswith("_at_or_above_30pct")}
        if (C.ACTS / tag / "anon_main_top1.u8.npy").exists():
            row["scale_and_base_rates"] = scale_block(tag)
        out[tag] = row
        print(tag, row["reference"]["tokens"], row.get("supplement", {}).get("tokens"))
    C.save(out, "passes.json")


if __name__ == "__main__":
    main()
