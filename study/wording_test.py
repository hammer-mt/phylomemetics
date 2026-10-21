"""Same idea, different words.

Two checks of whether meme neurons respond to an idea rather than to particular words:

1. Rewording. Each named meme's top sentence is rewritten so it shares no content words with the
   original (names replaced by descriptions). Does the Qwen neuron, and its Llama twin, still fire?
2. Revenge. Find each model's "revenge" neuron from the corpus sentences that use revenge words
   (revenge, vengeance, avenge, retribution, retaliate), choose among the top candidates on a dev set
   of revenge sentences that use none of those words, then test on fresh minimal pairs.

Activations are each unit's strongest response in the sentence, run on its own, reported as a
percentile of that unit's responses to all 3,700 corpus sentences.

Writes results/wording.json.
"""
import json
import os
import re
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from sklearn.feature_extraction.text import TfidfVectorizer

from discover import load_units
from extract import MODELS

ACTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("PHYLO_ACTS", "activations"))

REWORDED = {
    131948: "Orphaned as a child when gunmen shot his mother and father, the young man vowed to spend his life hunting criminals.",
    168352: "By day he was a harmless socialite; by night he hid his face behind a mask and fought for justice, and no one guessed the truth.",
    128514: "After his dad died, the young man inherited the family company and its vast wealth.",
    50224: "A wealthy loner secretly patrols the streets at night, punishing criminals the law cannot touch.",
    145011: "Beneath his mansion lies a hidden cavern full of computers, armored vehicles and grappling hooks.",
    105434: "A swarm of winged nocturnal creatures burst from the dark hollow, shrieking as they flew.",
    84130: "The crime-ridden metropolis was ruled by mobsters, its streets dark and its officials corrupt.",
    589605: "The city's honest police chief, a mustached veteran, met the caped hero on the rooftop beside the signal light.",
    584876: "The hero's young sidekick was beaten and left to die in an explosion, and the hero never forgave himself.",
    31044: "The clown delivered a punchline so dark that nobody in the audience laughed.",
    128822: "One after another he faced a mud monster, a madman in a top hat, a crocodile man and a psychiatrist obsessed with terror.",
    130069: "The alien child, rocketed away from a dying world, grew up able to fly and bend steel.",
    65066: "The video game includes dozens of familiar heroes and villains drawn from decades of the series, voiced by returning actors.",
    520801: "The family's loyal manservant raised the orphaned boy, cooking his meals and patching his wounds.",
    878889: "The rich bachelor adopted the circus acrobat whose mother and father had died, and took the boy into his home.",
    498697: "The heroes stormed the madman's hidden hideout, a frozen laboratory beneath an abandoned factory.",
    481199: "The device under the bridge exploded at noon, blowing debris across the river.",
    17909: "Mourners in black gathered by the open grave as the coffin was lowered into the earth.",
    448712: "Knocked off a catwalk, the gangster plunged into a tank of toxic waste and emerged disfigured and insane.",
    765185: "He let a flip decide whether his hostage lived or died, trusting chance alone, heads or tails.",
    68088: "The city's crusading district attorney, once its white knight, was scarred by acid and turned to crime.",
    13339: "The jewel thief in tight black leather slipped through the skylight and flirted with the masked hero.",
    465474: "The criminal taunted detectives with puzzles and cryptic clues, daring them to find the answers.",
    496024: "The second boy to wear the hero's mask was murdered by the clown, then came back years later wearing a red helmet.",
    501280: "The centuries-old leader of a league of assassins, kept young by a mystical pit, plotted to wipe out the metropolis.",
    513921: "The company's gadget engineer quietly gave the hero a tank-like car and a bulletproof suit from the research lab.",
    876522: "The new detective soon learned that most of his fellow officers took bribes from the mob.",
    57687: "The convicts broke out of their cells as the jail's doors swung open and they poured into the streets.",
    465435: "Poisoned and dying, he survived only because a friend synthesized a cure in time.",
    47532: "The experimental steroid turned the inmates into hulking monsters, but it slowly destroyed their bodies.",
    66576: "A gas that made victims see their worst nightmares sent the whole crowd into screaming panic.",
}

NEUTRAL = [
    "The bakery on the corner raised its prices for the second time this year.",
    "She watered the tomatoes before leaving for work.",
    "The committee approved the new parking rules on Tuesday.",
    "Rain is expected across the region through the weekend.",
    "He finally finished assembling the bookshelf after two hours.",
    "The museum extended its opening hours for the summer exhibition.",
    "Our flight was delayed because of a mechanical inspection.",
    "The students presented their science projects in the gym.",
    "A new coffee shop opened next to the library.",
    "The orchestra rehearsed the second movement three more times.",
]

REVENGE_WORDS = re.compile(r"\b(reveng\w*|vengean\w*|aveng\w*|vengeful|retribution|retaliat\w*)\b", re.I)

# Dev pairs: revenge without revenge words / the same sentence with the revenge removed.
DEV = [
    ("She spent ten years planning how to make her former partner suffer for betraying her.",
     "She spent ten years planning a surprise party for her former partner's retirement."),
    ("After they burned his village, he hunted down every soldier responsible, one by one.",
     "After the storm flattened his village, he rebuilt every house, one by one."),
    ("The fired employee sabotaged the company's servers to get back at his old boss.",
     "The new employee upgraded the company's servers to impress his boss."),
    ("He poisoned the rival chef's soup to repay him for the humiliation at the contest.",
     "He cooked the rival chef a bowl of soup to congratulate him on winning the contest."),
    ("They will pay for what they did to my brother, she whispered over the grave.",
     "We will miss everything he did for my brother, she whispered over the grave."),
    ("The jilted bride posted her ex-fiancé's secrets online so he would know how it felt.",
     "The bride posted photos of her fiancé online so everyone would know how happy she felt."),
    ("Years after the duel, the son tracked down his father's killer and challenged him.",
     "Years after the war, the son tracked down his father's old friend and thanked him."),
    ("To settle the score, the gang torched the warehouse of the family that had wronged them.",
     "To celebrate, the gang painted the warehouse of the family that had helped them."),
    ("He waited patiently in prison, knowing he would one day ruin the judge who sentenced him.",
     "He waited patiently in prison, knowing he would one day be released and see his daughter."),
    ("The bullied kid grew up and bought the company where his tormentor worked, just to fire him.",
     "The shy kid grew up and bought the company where his mentor worked, just to promote him."),
]
TEST = [
    ("The widow quietly ruined the businessman who had cheated her husband out of his savings.",
     "The widow quietly thanked the businessman who had helped her husband keep his savings."),
    ("He wanted the men who murdered his parents to feel the same pain he had felt.",
     "He wanted the nurses who cared for his parents to feel the same gratitude he had felt."),
    ("Humiliated in front of the whole school, she spent the summer plotting to destroy the girl who had mocked her.",
     "Inspired by the whole school, she spent the summer practicing to beat her own best time."),
    ("The pirate returned to the harbor that had hanged his crew and set every ship ablaze.",
     "The pirate returned to the harbor where he was born and bought every child a sweet."),
    ("Getting even was the only thing that kept him alive in that cell.",
     "Seeing his daughter again was the only thing that kept him alive in that cell."),
    ("An eye for an eye, the old man muttered, loading his rifle.",
     "Rain again tomorrow, the old man muttered, loading his truck."),
    ("She framed the detective who had locked away her innocent brother, sending him to the same prison.",
     "She helped the detective who had freed her innocent brother, sending him a thank-you letter."),
    ("The hacker leaked the CEO's emails to punish him for firing her team.",
     "The hacker fixed the CEO's emails to help him before the board meeting."),
    ("For every one of his men they killed, he swore to kill ten of theirs.",
     "For every one of his men they trained, he promised to train ten of theirs."),
    ("The ghost haunted the family whose ancestors had drowned her, one generation after another.",
     "The ghost watched over the family whose ancestors had loved her, one generation after another."),
    ("He made sure the bully who tormented him as a child lost his job, his house and his wife.",
     "He made sure the friend who protected him as a child got a job, a house and a party."),
    ("Payback came years later, when the double-crossed partner returned with a gun.",
     "Good news came years later, when the long-lost partner returned with a gift."),
]
SHOWCASE = [
    ("uses \"revenge\"", "He swore revenge on the men who killed his father."),
    ("uses \"vengeance\"", "He swore vengeance on the men who killed his father."),
    ("no revenge word", "He swore to make the men who killed his father pay for what they had done."),
    ("not revenge", "He swore to finish the race his father never got to run."),
]


class Grab(nn.Module):
    """Records each MLP's per-sentence max and min over tokens (first token excluded)."""

    def __init__(self, mlp, sink, idx):
        super().__init__()
        self.mlp, self._sink, self._idx = mlp, sink, idx

    def __call__(self, x):
        h = nn.silu(self.mlp.gate_proj(x)) * self.mlp.up_proj(x)
        hs = h[0, 1:]
        self._sink["pos"][self._idx] = hs.max(axis=0)
        self._sink["neg"][self._idx] = hs.min(axis=0)
        return self.mlp.down_proj(h)


def run_sentences(tag, sentences):
    """[n_sentences, 2N] unit activations, positive pole then negative pole (negated), like the corpus."""
    model, tok = load(MODELS[tag])
    layers = model.model.layers
    sink = {"pos": [None] * len(layers), "neg": [None] * len(layers)}
    for i, layer in enumerate(layers):
        layer.mlp = Grab(layer.mlp, sink, i)
    out = []
    for s in sentences:
        ids = tok._tokenizer(s, add_special_tokens=True)["input_ids"]
        model.model(mx.array([ids]))
        p, q = mx.concatenate(sink["pos"]), mx.concatenate(sink["neg"])
        mx.eval(p, q)
        out.append(np.concatenate([np.array(p.astype(mx.float32)), -np.array(q.astype(mx.float32))]))
    del model
    mx.clear_cache()
    return np.stack(out)


def corpus_profiles(tag, units):
    """Sentence-level corpus activations [3700, len(units)] for the given units, sign-matched."""
    meta, pos, neg, ps, ns = load_units(ACTS / tag)
    n = pos.shape[1]
    cols = []
    for u in units:
        cols.append(pos[:, u].astype(np.float32) * ps[u] if u < n else -neg[:, u - n].astype(np.float32) * ns[u - n])
    return np.stack(cols, axis=1)


def pct(corpus_col, v):
    return float((corpus_col < v).mean() * 100)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def revenge_neuron(tag, works, sentences):
    meta, pos, neg, ps, ns = load_units(ACTS / tag)
    n = pos.shape[1]
    is_rev = np.array([bool(REVENGE_WORDS.search(s)) for s in sentences])
    # effect size of each unit on revenge-word sentences vs all others, in chunks
    best = []
    for arr, sc, sign, off in ((pos, ps, 1, 0), (neg, ns, -1, n)):
        for c in range(0, n, 65536):
            X = sign * arr[:, c:c + 65536].astype(np.float32) * sc[c:c + 65536]
            mu_r, mu_o, sd_o = X[is_rev].mean(0), X[~is_rev].mean(0), X[~is_rev].std(0) + 1e-6
            d = (mu_r - mu_o) / sd_o
            top = np.argsort(-d)[:50]
            best += [(float(d[j]), off + c + int(j)) for j in top]
    best.sort(reverse=True)
    cands = [u for _, u in best[:50]]
    dev_r, dev_c = [r for r, _ in DEV], [c for _, c in DEV]
    test_r, test_c = [r for r, _ in TEST], [c for _, c in TEST]
    show = [s for _, s in SHOWCASE]
    A = run_sentences(tag, dev_r + dev_c + test_r + test_c + show)
    k = len(DEV)
    Adr, Adc = A[:k], A[k:2 * k]
    At_r, At_c = A[2 * k:2 * k + len(TEST)], A[2 * k + len(TEST):2 * k + 2 * len(TEST)]
    As = A[2 * k + 2 * len(TEST):]
    dev_auc = [auc(Adr[:, u], Adc[:, u]) for u in cands]
    pick = cands[int(np.argmax(dev_auc))]
    top_kw = cands[0]
    prof = corpus_profiles(tag, [pick, top_kw])
    order = np.argsort(-prof[:, 0])[:6]

    def describe(u, col):
        return {
            "unit": int(u), "layer": int((u % n) // meta["hidden"]), "neuron": int((u % n) % meta["hidden"]),
            "pole": "-" if u >= n else "+",
            "dev_auc": round(dev_auc[cands.index(u)], 3),
            "test_auc": round(auc(At_r[:, u], At_c[:, u]), 3),
            "test_pairs_won": int((At_r[:, u] > At_c[:, u]).sum()),
            "test_revenge_pct": [round(pct(prof[:, col], v), 1) for v in At_r[:, u]],
            "test_control_pct": [round(pct(prof[:, col], v), 1) for v in At_c[:, u]],
            "showcase": [{"kind": kind, "text": s, "pct": round(pct(prof[:, col], As[i, u]), 1)}
                         for i, (kind, s) in enumerate(SHOWCASE)],
        }

    res = {"n_keyword_sentences": int(is_rev.sum()), "n_candidates": len(cands),
           "chosen": describe(pick, 0), "top_keyword_unit": describe(top_kw, 1),
           "chosen_top_corpus": [{"work": works[meta["sent_work"][i]]["label"], "text": sentences[i]} for i in order]}
    return res, pick


def main():
    works = [json.loads(l) for l in open("data/corpus.jsonl")]
    sentences = [s for w in works for s in w["sentences"]]
    labels = json.load(open("results/labels.json"))
    fams = json.load(open("results/qwen3-8b/memes.json"))["families"]
    fidx = {f["unit"]: i for i, f in enumerate(fams)}
    phylo = json.load(open("results/phylo.json"))
    twins = phylo["models"]["llama3.1-8b"]["partners"]

    units_q = [int(u) for u in labels]
    originals = {u: fams[fidx[u]]["top_sentences"][0]["text"] for u in units_q}
    units_l = [twins[fidx[u]]["unit"] for u in units_q]

    # word overlap between each original and its rewording, using the corpus TF-IDF vocabulary
    tfidf = TfidfVectorizer(stop_words="english", min_df=1, sublinear_tf=True).fit(sentences + list(REWORDED.values()))
    ov = {u: float((tfidf.transform([originals[u]]) @ tfidf.transform([REWORDED[u]]).T).toarray()[0, 0]) for u in units_q}

    batch = [originals[u] for u in units_q] + [REWORDED[u] for u in units_q] + NEUTRAL
    out = {"memes": [], "neutral": NEUTRAL}
    acts = {}
    for tag, units in (("qwen3-8b", units_q), ("llama3.1-8b", units_l)):
        A = run_sentences(tag, batch)
        prof = corpus_profiles(tag, units)
        m = len(units)
        acts[tag] = [{"original": pct(prof[:, j], A[j, u]), "reworded": pct(prof[:, j], A[m + j, u]),
                      "neutral_max": pct(prof[:, j], A[2 * m:, u].max())} for j, u in enumerate(units)]
    for j, u in enumerate(units_q):
        out["memes"].append({"unit": u, "label": labels[str(u)]["label"], "original": originals[u],
                             "reworded": REWORDED[u], "word_overlap": round(ov[u], 3),
                             "qwen": {k: round(v, 1) for k, v in acts["qwen3-8b"][j].items()},
                             "llama": {k: round(v, 1) for k, v in acts["llama3.1-8b"][j].items()},
                             "llama_r": twins[fidx[u]]["r"]})
        q, l = acts["qwen3-8b"][j], acts["llama3.1-8b"][j]
        print(f"{labels[str(u)]['label'][:34]:34s} overlap {ov[u]:.2f} | qwen orig {q['original']:5.1f} reworded {q['reworded']:5.1f} neutral {q['neutral_max']:5.1f} | llama orig {l['original']:5.1f} reworded {l['reworded']:5.1f} neutral {l['neutral_max']:5.1f}")

    out["revenge"] = {}
    picks = {}
    for tag in ("qwen3-8b", "llama3.1-8b"):
        res, picks[tag] = revenge_neuron(tag, works, sentences)
        out["revenge"][tag] = res
        c = res["chosen"]
        print(tag, "revenge neuron", f"L{c['layer']}·{c['neuron']}{c['pole']}", "dev AUC", c["dev_auc"], "test AUC", c["test_auc"],
              "pairs won", c["test_pairs_won"], "/", len(TEST), "| top keyword unit test AUC", res["top_keyword_unit"]["test_auc"])
        for s in c["showcase"]:
            print("   ", s["kind"], s["pct"])
    # do the two models' revenge neurons fire on the same corpus sentences?
    pq = corpus_profiles("qwen3-8b", [picks["qwen3-8b"]])[:, 0]
    pl = corpus_profiles("llama3.1-8b", [picks["llama3.1-8b"]])[:, 0]
    out["revenge"]["cross_model_r"] = round(float(np.corrcoef(pq, pl)[0, 1]), 3)
    print("revenge neurons, Qwen vs Llama, corpus correlation:", out["revenge"]["cross_model_r"])
    out["revenge"]["raw"] = revenge_raw(out["revenge"])
    json.dump(out, open("results/wording.json", "w"), indent=1)


def revenge_raw(rev):
    """Raw activations of each model's chosen revenge neuron on the showcase sentences and test pairs."""
    raw = {}
    for tag in ("qwen3-8b", "llama3.1-8b"):
        u = rev[tag]["chosen"]["unit"]
        sents = [s for _, s in SHOWCASE] + [r for r, _ in TEST] + [c for _, c in TEST]
        A = run_sentences(tag, sents)[:, u]
        corpus = corpus_profiles(tag, [u])[:, 0]
        T = len(TEST)
        raw[tag] = {"showcase": [round(float(v), 3) for v in A[:4]],
                    "test_revenge": [round(float(v), 3) for v in A[4:4 + T]],
                    "test_control": [round(float(v), 3) for v in A[4 + T:]],
                    "corpus_p50": round(float(np.median(corpus)), 3), "corpus_p99": round(float(np.percentile(corpus, 99)), 3)}
    return raw


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "raw-only":
        out = json.load(open("results/wording.json"))
        out["revenge"]["raw"] = revenge_raw(out["revenge"])
        json.dump(out, open("results/wording.json", "w"), indent=1)
    else:
        main()
