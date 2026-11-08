#!/bin/bash
# On the GPU machine: every pass on the tiny test model, in the precision and on the device the real runs use, then the
# pipeline test. It reads the first 4 study works only (as the CPU test does) and writes under /workspace/out_test.
set -uo pipefail
cd /workspace/code
T=/workspace/out_test
rm -rf $T && mkdir -p $T/study
head -4 /workspace/data/study/corpus.jsonl > $T/study/corpus.jsonl
head -4 /workspace/data/study/corpus_neutral.jsonl > $T/study/corpus_neutral.jsonl
Q='Loading weights\|Fetching\|Warning'
python3 ref_pass.py qwen3-0.6b /workspace/data/reference.jsonl $T --limit-docs 48 --batch-tokens 4096 2>&1 | grep -v "$Q" | tail -2
python3 ref_pass.py qwen3-0.6b /workspace/data/supplement.jsonl $T --limit-docs 24 --batch-tokens 4096 --name ref_sup --max-tokens 2048 2>&1 | grep -v "$Q" | tail -1
python3 study_pass.py qwen3-0.6b $T/study $T 2>&1 | grep -v "$Q" | tail -1
python3 finalize_scale.py qwen3-0.6b $T | head -6
python3 -c "import numpy as np; np.save('$T/sel.npy', np.array([5, 1000, 86023, 136016, 30737]))"
python3 background_pass.py qwen3-0.6b /workspace/data/reference.jsonl $T --select $T/sel.npy --limit-docs 6 --exclude-docs 843 2>&1 | grep -v "$Q" | tail -1
python3 background_pass.py qwen3-0.6b /workspace/data/franchises.jsonl $T --src franchise --name franchises --limit-docs 4 2>&1 | grep -v "$Q" | tail -1
python3 study_sentences.py qwen3-0.6b $T/study $T $T/sel.npy 2>&1 | grep -v "$Q" | tail -1
printf '%s\n' '{"id": "a", "text": "After his parents are murdered in an alley, the boy swears revenge."}' '{"id": "b", "text": "She bought milk."}' '{"id": "c", "text": "Batman glides over the city."}' > $T/qwen3-0.6b/score_test.jsonl
python3 score_texts.py qwen3-0.6b $T/qwen3-0.6b/score_test.jsonl $T/sel.npy $T/qwen3-0.6b/score_test.npz 2>&1 | grep -v "$Q" | tail -1
python3 top_examples.py qwen3-0.6b /workspace/data/reference.jsonl,/workspace/data/supplement.jsonl $T $T/sel.npy $T/top.json --k 6 2>&1 | grep -v "$Q" | tail -1
python3 steer.py qwen3-0.6b $T $T/sel.npy $T/steer.json --multiples 1 3 --tokens 12 2>&1 | grep -v "$Q" | tail -1
python3 - <<'PY'
import json, numpy as np
T = "/workspace/out_test"
works = [json.loads(l) for l in open(T + "/study/corpus_neutral.jsonl")]
rng = np.random.default_rng(0)
units = rng.choice(2 * 28 * 3072, 200, replace=False)
np.savez(T + "/partners.npz", units=units, partners=rng.choice(2 * 28 * 3072, (3, 200)), partners_distinctive=rng.choice(2 * 28 * 3072, (2, 200)))
items = [{"id": w["label"], "text": " ".join(w["sentences"][:6]), "answers": ["Batman", "The Batman"], "members": list(range(0, 200, 2)), "conditions": ["canon", "rand:1", "distinctive:0"]} for w in works[:2]]
for mode in ("all", "strong"):
    json.dump({"question": "Who is the hero? Reply with only the name.", "partner_file": T + "/partners.npz", "reply_for": ["canon"], "mode": mode, "items": items}, open(f"{T}/ablate_{mode}.json", "w"))
PY
python3 ablate.py qwen3-0.6b $T/ablate_all.json $T/ablate_all_out.json 2>&1 | grep -v "$Q" | tail -1
python3 ablate.py qwen3-0.6b $T/ablate_strong.json $T/ablate_strong_out.json $T/qwen3-0.6b/study/scale.npy 2>&1 | grep -v "$Q" | tail -1
python3 twin_search.py --help > /dev/null 2>&1 && echo "twin_search imports"
python3 test_pipeline.py qwen3-0.6b /workspace/data/reference.jsonl $T/study $T $T/sel.npy 2>&1 | grep -v "$Q" | tail -48
echo SMOKE_DONE
