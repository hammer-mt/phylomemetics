"""Write scout/manifest.json: every scout output file with its size, row count and what it holds."""
import json
import os
from pathlib import Path

S = Path("<scratchpad>/prereg/scout")
DESC = {
    "plays.jsonl": "40 Shakespeare plays: plot/synopsis section (fetch_corpus.py rule), sentences, principal names (raw text, not analysed)",
    "category_members.json": "Shakespeare film/TV category tree crawl (46 categories)",
    "films_all.jsonl": "every article in the Shakespeare category tree, named extras and Wikidata P144/P941 derivatives, with plot text and attribution metadata",
    "shakespeare_wikidata.json": "Wikidata P144 based-on / P941 inspired-by links to each play",
    "shakespeare_adaptations.jsonl": "per-article selection metadata: documented sources and evidence type, is_film, names kept, source mentions, DEV overlap, eligible/primary flags (no text)",
    "shakespeare_summary.json": "counts for T1d",
    "lang_check_qwen3-8b.json": "made-up sentences: token cost, spans, back-translation, continuation (zh, ja, ru, hi)",
    "lang_check_llama3.1-8b.json": "same for Llama",
    "lang_check2_qwen3-8b.json": "made-up passage: translation and 3-question QA (zh, ja, ru, hi)",
    "lang_check2_llama3.1-8b.json": "same for Llama",
    "lang_check3_qwen3-8b.json": "made-up text: es/de/fr/th token cost and spans; 24-sentence story QA in en/zh/es",
    "lang_check3_llama3.1-8b.json": "same for Llama",
    "embed_arch_check.json": "earlier encoder architecture check (3 models)",
    "t3/category_members.json": "T3 category crawl: 21 roots, buckets and category paths per article",
    "t3/candidates.jsonl": "every T3 candidate article: metadata plus raw plot section and sentences (not analysed)",
    "t3/random_population.json": "random-film population: direct members of Category:YYYY films 1915-2024 with year",
    "t3/random_films.jsonl": "hash-ordered walk of the population; accepted films carry raw plot text",
    "t3/pool_summary.json": "T3 counts by bucket, kind, decade; random-sample counts",
    "t3/pool_eligible.jsonl": "eligible T3 pool works (metadata only)",
    "relatives/wikidata.json": "Wikidata P941/P144/P4969 links into and out of Batman",
    "relatives/wikipedia_evidence.json": "influence sentences from Batman-related articles; full-text search hits for derivative phrases",
    "relatives/documented_relatives.json": "curated ground truth: relation, strength, evidence sentences, candidate works with plot sizes",
    "relatives/works.jsonl": "fetched works of documented relatives with raw plot text",
    "tools/meta.json": "HF model metadata (sha, licence, weight size) and PyPI metadata for packages not installed",
    "tools/timing.json": "dummy-input timings: TF-IDF, presence, LSA, LDA, PCA+HDBSCAN, k-means, c-TF-IDF, 8 encoders (random init)",
    "tools/llm_timing_qwen3-8b.json": "made-up story: judge per meme (no cache, cached prefix), multi-meme call, flat and recursive summaries",
    "tools/llm_timing_llama3.1-8b.json": "same for Llama",
    "tools/neuron_timing_qwen3-8b.json": "made-up story: forward + per-work top-3 aggregate on GPU vs full copy",
    "tools/neuron_timing_llama3.1-8b.json": "same for Llama",
}
out = {}
for rel, d in DESC.items():
    p = S / rel
    if not p.exists():
        out[rel] = dict(missing=True, what=d)
        continue
    n = sum(1 for _ in open(p)) if rel.endswith(".jsonl") else None
    out[rel] = dict(path=str(p), bytes=os.path.getsize(p), rows=n, what=d)
out["raw/pages/"] = dict(path=str(S / "raw" / "pages"), files=len(os.listdir(S / "raw" / "pages")), what="cached Wikipedia API responses (extract, categories, pageprops)")
out["scripts"] = dict(path=str(S.parent / "scripts"), files=sorted(os.listdir(S.parent / "scripts")))
json.dump(out, open(S / "manifest.json", "w"), indent=1)
print(json.dumps(out, indent=1))
