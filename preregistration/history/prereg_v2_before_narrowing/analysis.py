"""Frozen analysis for the phylomemetics pre-registration (prereg-v2, revised 2026-10-21 after four AI design reviews).

Organising question: when, and at what cost, do neuron-defined memes beat keywords, vectors, topics and LLM reading?
Every test runs the same methods on the same inputs. This file holds every statistic, every cheap text method (TF-IDF,
word presence, LSA, LDA, embedding-clustering topics, lexical probes, keyword queries), every detector fitted on DEV,
every test runner, the cost model and the verdict rules. It imports no other project module.

Model-side features come from three other frozen scripts in this folder:
  extract_compact.py  neuron aggregates (per-work top-3 mean for every unit), per-sentence values for a fixed unit list,
                      per-sentence mean hidden states
  readers.py          sentence, window and document embeddings; LLM judges (full-question, shared-sheet, one-pass, small
                      model), NLI, recursive summaries, LLM lineage reading (summaries and full texts), likeness,
                      recognition probe, keyword choice (plain and evasion-aware), translation, cross-family check reader
  bench.py            controlled cost benchmark on DEV texts
They write into prereg/features/<set>/ using the FEATURE CONTRACTS below.

Usage (from the project root, with .venv; the order of work is PREREGISTRATION.md section 3):
  python prereg/analysis.py dev-freeze      # freeze F1 (DEV only) -> prereg/dev_freeze.json, sentence_units_<model>.npy
  python prereg/analysis.py t6-prepare      # T6 label-test items from the T3 pool features
  python prereg/analysis.py t1 | t2 | t3 | t4 | t5 | t6 | cost
  python prereg/analysis.py primary         # H1-H5 with Holm and three-way verdicts, H6, Llama replication
  python prereg/analysis.py manifest        # freeze F3: sha256 of every TEST feature file, before any test runner
Outputs go to prereg/results/<test>.json.

FEATURE CONTRACTS (all under prereg/features/<set>/, where <set> names prereg/data/<set>.jsonl, one work per line:
{"id", "sentences": [...], ...}; sentence order = works in file order, sentences in work order):
  neurons_<model>/agg.npy     float16 [n_works, n_units]  mean of each work's 3 strongest sentences per unit (unit u < N:
                              max over the sentence's tokens of SwiGLU channel u; u >= N: minus the min of channel u - N)
  neurons_<model>/sent.npy    float16 [n_sentences, k]    per-sentence values for units.npy (same sign rule)
  neurons_<model>/units.npy   int64 [k]
  neurons_<model>/full.npy    float16 [n_sentences, n_units] (small DEV sets only)
  neurons_<model>/hidden.npy  float16 [n_sentences, n_layers, d] mean hidden state over the sentence's tokens
  neurons_<model>/meta.json   {"work_ids", "sent_work", "n_units", "hidden_layers", "set_sha256", "seconds", ...}
  emb_<enc>.npy / emb_<enc>_win3.npy   float16 [n_sentences, d] L2-normalised sentence / 3-sentence-window embeddings
  emb_<enc>_doc.npy           float16 [n_works, d]        whole-work embeddings
  llm_<reader>_<task>.jsonl   judge, judge-shared, judge-1pass, sjudge: {"work", "meme", "p_yes", "logit"};
                              summary: {"work", "summary"}; pairs, pairsfull: {"a", "b", "p_yes", "logit"};
                              likeness: {"work", "p_yes", "logit"}; recognise: {"work", "answer"}
  nli_<model>.npy             float16 [n_sentences, n_memes] entailment probability per sentence and meme
  cost_<method>.json          {"seconds", "peak_gb", "bytes", "calls", "tokens", "load_seconds", "loadavg"}
"""
import argparse
import difflib
import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sps

VERSION = "prereg-v2"
PRE = Path(__file__).resolve().parent
ROOT = PRE.parent
DATA = Path(os.environ.get("PREREG_DATA", PRE / "data"))       # overridable for tests
FEAT = Path(os.environ.get("PREREG_FEATURES", PRE / "features"))
RES = Path(os.environ.get("PREREG_RESULTS", PRE / "results"))
SEED = 20261020
ALPHA = 0.05
TOPK = 3                      # a work's score on a unit = mean of its TOPK strongest sentences
N_PERM_LINEAGE = 20000        # time-consistent joint-relabelling permutations for every lineage test
N_SIGNFLIP_MC = 200000        # Monte Carlo sign flips when exact enumeration is too large
EXACT_SIGNFLIP_MAX = 2 ** 21  # enumerate every sign pattern up to this many
N_BOOT = 10000
NI_MARGIN = 0.05              # H2 non-inferiority margin in AUC
SESOI = {"H1": 0.05, "H2": 0.05, "H3": 0.05, "H4": 0.05, "H5": 0.05}   # smallest effects of interest (AUC / share)
HUB_SD_FLOOR_Q = 0.10         # hub SDs are floored at this quantile of all column SDs
MODELS = ["qwen3-8b", "llama3.1-8b"]
PRIMARY_MODEL = "qwen3-8b"
SMALL_READER = "qwen3-1.7b"
ENCODERS = {"qwen3emb-4b": "Qwen/Qwen3-Embedding-4B", "e5-large": "intfloat/multilingual-e5-large",
            "e5-base": "intfloat/multilingual-e5-base", "labse": "sentence-transformers/LaBSE",
            "minilm": "sentence-transformers/all-MiniLM-L6-v2"}
PRIMARY_ENCODER = "qwen3emb-4b"         # the strongest encoder that fits this laptop, fixed a priori
H1_RIVALS = ["keywords:expanded-qwen3-8b", "keywords:logreg"]          # intersection-union
H2_RIVALS = ["emb:qwen3emb-4b", "embwin:qwen3emb-4b", "emb:e5-large"]  # intersection-union
LINEAGE_RIVALS = ["tfidf", "lsa"]                                      # intersection-union (H4, H5)
DRIFT_LEVELS = [0, 25, 50, 75, 100]
H4_LEVELS = [50, 75, 100]
H4_CONJUNCT_LEVELS = [75, 100]
VERSIONS = ["explicit", "banned", "implicit", "coded", "control", "decoy"]
PLANTED = {"explicit", "banned", "implicit", "coded"}
REWORDED = {"banned", "implicit"}
CONDITIONS = {"reworded": REWORDED, "explicit": {"explicit"}, "banned": {"banned"}, "implicit": {"implicit"}, "coded": {"coded"}}
EXAMPLE_SETS = {"main": ("definition", "explicit", "reworded"), "explicit": ("definition", "explicit"), "far": ("definition", "far")}
POSITIVE_ROWS = ("definition", "explicit", "reworded", "far")
TIMING = defaultdict(dict)    # analysis-time cost of the cheap methods: TIMING[set][method] = seconds


# =============================================================================================== io
def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=_jsonable)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(type(o))


def sha256_text(sentences):
    return hashlib.sha256("\n".join(sentences).encode("utf-8")).hexdigest()


def sha256_file(path):
    hh = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            hh.update(b)
    return hh.hexdigest()


def rng_for(*keys):
    """A generator seeded from SEED and a key, so every test has its own reproducible stream."""
    hh = hashlib.sha256(("|".join(map(str, (SEED,) + keys))).encode()).digest()
    return np.random.default_rng(int.from_bytes(hh[:8], "little"))


def timed(store, key):
    """Context manager that adds the elapsed seconds to store[key]."""
    class _T:
        def __enter__(self):
            self.t0 = time.time()
        def __exit__(self, *a):
            store[key] = store.get(key, 0.0) + time.time() - self.t0
    return _T()


# =============================================================================================== text
try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _SKSTOP
    STOP = frozenset(_SKSTOP)
except Exception:  # pragma: no cover
    STOP = frozenset()
CODE = re.compile(r"\bX\d+\b")
WORD = re.compile(r"[a-z][a-z'\-]*[a-z]|[a-z]")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
CJK = re.compile(r"[㐀-鿿豈-﫿]")


def is_cjk(text):
    return bool(CJK.search(text))


def _undouble(w):
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiouslz":
        return w[:-1]
    return w


def stem(w):
    """Light, fixed English stemmer used everywhere a word is matched (keywords, ban lists, drift types)."""
    w = w.lower().replace("’", "'")
    w = re.sub(r"'s?$", "", w)
    if len(w) <= 3:
        return w
    if w.endswith(("ies", "ied")) and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("es") and w[-3:-2] in ("s", "x", "z") or w.endswith(("ches", "shes")):
        w = w[:-2]
    elif w.endswith("ing") and len(w) > 5:
        w = _undouble(w[:-3])
    elif w.endswith("ed") and len(w) > 4:
        w = _undouble(w[:-2])
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
        w = w[:-1]
    if w.endswith("e") and len(w) >= 4:
        w = w[:-1]
    return w


def tokens(text):
    return WORD.findall(CODE.sub(" ", text).lower().replace("’", "'"))


def stems(text):
    return [stem(t) for t in tokens(text)]


def content_types(sentences, digits=False):
    """Content-word types of a text: stems of alphabetic tokens of >= 3 letters that are not stop words; with
    digits=True also every number (drift bans numbers too, so issue numbers cannot carry the lineage)."""
    out = set()
    for s in sentences:
        for t in tokens(s):
            if len(t) >= 3 and t not in STOP and t.replace("-", "").replace("'", "").isalpha():
                out.add(stem(t))
        if digits:
            out |= {"#" + n for n in NUMBER.findall(CODE.sub(" ", s))}
    return out


def term_stems(term):
    return [stem(t) for t in WORD.findall(term.lower().replace("’", "'"))]


def _content(ts):
    c = [t for t in ts if t not in STOP]
    return c or ts


def keyword_hits(sentence, terms):
    """Search score of a sentence for a keyword query (every keyword method). A one-word term counts each occurrence
    of its stem; a multi-word term counts 1 when all its content-word stems occur in the sentence. Chinese terms count
    substring occurrences. The score is the sum over terms."""
    if not terms:
        return 0
    st, sset, n = None, None, 0
    for term in terms:
        if is_cjk(term):
            n += sentence.count(term)
            continue
        if st is None:
            st = stems(sentence)
            sset = set(st)
        ts = term_stems(term)
        if not ts:
            continue
        if len(ts) == 1:
            n += st.count(ts[0])
        else:
            n += int(all(t in sset for t in _content(ts)))
    return n


class StemCache:
    """Stems of every sentence of a set, computed once."""

    def __init__(self, sentences):
        self.sentences = sentences
        self.st = [stems(s) for s in sentences]
        self.sets = [set(x) for x in self.st]

    def hits(self, i, terms):
        n = 0
        for term in terms:
            if is_cjk(term):
                n += self.sentences[i].count(term)
                continue
            ts = term_stems(term)
            if not ts:
                continue
            if len(ts) == 1:
                n += self.st[i].count(ts[0])
            else:
                n += int(all(t in self.sets[i] for t in _content(ts)))
        return n


def phrase_hits(sentence, terms):
    """Exact phrase occurrences (consecutive stems, stop words kept)."""
    st = stems(sentence)
    n = 0
    for term in terms:
        if is_cjk(term):
            n += sentence.count(term)
            continue
        ts = term_stems(term)
        k = len(ts)
        if k:
            n += sum(1 for i in range(len(st) - k + 1) if st[i:i + k] == ts)
    return n


# derivational families of short stems that the prefix rule cannot reach (frozen; review round 1)
BAN_FAMILY_WORDS = {"kill": ["killer", "killers"], "bribe": ["bribery", "briber"], "fear": ["fearful", "fearsome", "fearless"],
                    "luck": ["lucky", "unlucky", "luckily"], "cure": ["curative"], "dose": ["dosage"], "heir": ["heiress", "heirloom"],
                    "mask": ["unmask", "masquerade"], "cop": ["copper"], "rich": ["richly", "richer"], "ward": ["wardship"],
                    "door": ["doorway", "doorstep"], "late": ["latecomer"], "shoe": ["shoemaker"], "vow": ["avow"],
                    "fate": ["fateful", "fatalism"], "gas": ["gassy"], "grin": ["grinning"], "toss": ["tossup"]}


def _common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


BAN_FAMILIES = {}


def _ban_families():
    if not BAN_FAMILIES:
        for k, v in BAN_FAMILY_WORDS.items():
            BAN_FAMILIES[stem(k)] = {stem(w) for w in v}
    return BAN_FAMILIES


def ban_hits(sentence, terms):
    """Ban-list matching for the data-generation checks (Q2 and D2), stricter than any method: a multi-word term
    matches as an exact phrase; a one-word term matches any token whose stem equals its stem, shares a prefix of at
    least 5 letters with a stem of 5 or more letters (avenge/avenger, funeral/funerary, orphan/orphanage), or is in the
    frozen derivational family of a short stem (kill/killer, bribe/bribery)."""
    st = stems(sentence)
    n = 0
    for term in terms:
        if is_cjk(term):
            n += sentence.count(term)
            continue
        ts = term_stems(term)
        if not ts:
            continue
        if len(ts) > 1:
            n += phrase_hits(sentence, [term])
            continue
        s = ts[0]
        fam = _ban_families().get(s, set())
        for t in st:
            if t == s or t in fam or (len(s) >= 5 and len(t) >= 5 and _common_prefix(s, t) >= 5):
                n += 1
    return n


def banned_terms_present(sentences, terms):
    return sorted({t for t in terms for s in sentences if ban_hits(s, [t])})


def types_banned(types_present, banned):
    """D2 with the derivational rule: which present content types fall under a banned type."""
    out = set()
    for t in types_present:
        if t.startswith("#"):
            if t in banned:
                out.add(t)
            continue
        for b in banned:
            if b.startswith("#"):
                continue
            if t == b or t in _ban_families().get(b, set()) or (len(b) >= 5 and len(t) >= 5 and _common_prefix(b, t) >= 5):
                out.add(t)
                break
    return out


def n_words(sentences):
    return sum(len(s.split()) for s in sentences)


# ---------------------------------------------------------------- deterministic name masker and T3 text rule
_CONNECT = {"of", "the", "de", "von", "van", "al", "bin", "la", "le", "du", "da", "del", "di"}
_HONOR = {"Mr.", "Mrs.", "Dr.", "St.", "Jr.", "Sr.", "Ms.", "Lt.", "Sgt.", "Gen.", "Capt.", "Det.", "Prof.", "Sir", "Lady", "Lord"}
_LEAD = re.compile(r"^[(\"“‘'\[]*")
_TAIL = re.compile(r"[.,;:!?)\"”’'\]]*$")
_POSS = re.compile(r"(['’]s|['’])$")


def _core(tok):
    return _POSS.sub("", _TAIL.sub("", _LEAD.sub("", tok)))


def mask_names(sentences):
    """Replace capitalised name spans with X-codes, consistent within a work (X1, X2 ... by first appearance).
    A capitalised token is masked when it is not sentence-initial, or when the same word also appears capitalised
    in a non-initial position elsewhere in the work, or when it is an honorific. Connectors between two masked tokens
    join the span. On the DEV corpus this masks 91% of the tokens the agent-made name-free texts replaced, at 97%
    precision; 129 of 13,337 agent-replaced tokens are capitalised words it misses (hence the FRANCHISE scan)."""
    toks = [s.split() for s in sentences]
    mid = Counter()
    for t in toks:
        for i, w in enumerate(t):
            c = _core(w)
            if i > 0 and c[:1].isupper() and not t[i - 1].endswith((".", "!", "?", ":", "\"")):
                mid[c] += 1
    codes, part_of, out, n_spans = {}, {}, [], 0
    for t in toks:
        flags = []
        for i, w in enumerate(t):
            c = _core(w)
            if not c[:1].isupper() or c == "I":
                flags.append(False)
                continue
            start = i == 0 or t[i - 1].endswith((".", "!", "?", ":"))
            flags.append((not start) or mid[c] > 0 or w in _HONOR)
        for i in range(1, len(t) - 1):
            if not flags[i] and _core(t[i]).lower() in _CONNECT and flags[i - 1] and flags[i + 1]:
                flags[i] = True
        new, i = [], 0
        while i < len(t):
            if not flags[i]:
                new.append(t[i])
                i += 1
                continue
            j = i
            while j + 1 < len(t) and flags[j + 1] and not re.search(r"[.,;:!?)\"”]$", t[j]):
                j += 1
            span = t[i:j + 1]
            key = re.sub(r"[^\w\s-]", "", " ".join(_core(w) for w in span)).strip().lower()
            if key not in codes and " " not in key and part_of.get(key):
                codes[key] = part_of[key]
            code = codes.setdefault(key, f"X{len(set(codes.values())) + 1}")
            if " " in key:
                for part in key.split():
                    if part not in _CONNECT:
                        part_of[part] = part_of.get(part) if part_of.get(part, code) != code else code
            lead = _LEAD.match(span[0]).group(0)
            last = span[-1]
            poss = "'s" if re.search(r"['’]s\W*$", last) else ""
            tail = _TAIL.search(_POSS.sub("", last) if poss else last).group(0)
            new.append(f"{lead}{code}{poss}{tail}")
            n_spans += 1
            i = j + 1
        out.append(" ".join(new))
    return out, n_spans


FRANCHISE = re.compile(r"\b(batman|gotham|wayne|joker|riddler|penguin|alfred|arkham|batmobile|batcave|two-face|catwoman|"
                       r"harley|zorro|pimpernel|stark|spider-man|daredevil|wick|dick grayson|jason todd|gordon)\b", re.I)
FRANCHISE_ZH = re.compile(r"蝙蝠侠|哥谭|高谭|韦恩|罗宾|阿尔弗雷德|阿卡姆|双面人|猫女|小丑女|蝙蝠车|蝙蝠洞|佐罗|红花侠|钢铁侠|蜘蛛侠")
RELATION_STATEMENT = re.compile(r"parod(y|ies|ied)\b|based on|inspired by|homage to|pastiche", re.I)


def strip_relation_statements(sentences):
    """T3 text rule (applied to every pool work and the DEV reference alike): parentheticals that state a relation
    ('a parody of X', 'based on', 'inspired by', 'homage to', 'pastiche') are removed; a sentence that still states one
    is dropped. Returns (sentences, n_removed)."""
    out, n = [], 0
    for s in sentences:
        t = re.sub(r"\s*\([^()]*\)", lambda m: "" if RELATION_STATEMENT.search(m.group(0)) else m.group(0), s)
        if t != s:
            n += 1
        if RELATION_STATEMENT.search(t):
            n += 1
            continue
        out.append(t)
    return out, n


def t3_text(sentences):
    """T3 and T2-N text: relation statements stripped, names masked, then any surviving franchise word masked."""
    s, n_rel = strip_relation_statements(sentences)
    m, n_spans = mask_names(s)
    hits = Counter()
    out = []
    for x in m:
        for mt in FRANCHISE.finditer(x):
            hits[mt.group(0).lower()] += 1
        out.append(FRANCHISE.sub("X0", x))
    return out, {"n_relation_removed": n_rel, "n_spans_masked": n_spans, "franchise_hits": dict(hits)}


def collapse_codes(sentences):
    """X-code sensitivity: every code becomes the one token X0, so cast size and code positions carry nothing."""
    return [CODE.sub("X0", s) for s in sentences]


def neutralise(original, rewrite):
    """name_removal_protocol.md step 3 (copied from make_neutral.py): the agent rewrite is used only to FIND names.
    A word-level diff against the original marks every changed span; each is replaced by a neutral code (X1, X2 ...
    in order of first appearance, the same code for the same original words). Deleted spans are deleted."""
    def key(words):
        return re.sub(r"[^\w\s-]", "", re.sub(r"'s\b|’s\b", "", " ".join(words))).strip().lower()
    codes, new_sents, n_spans = {}, [], 0
    for a, b in zip(original, rewrite):
        A_, B_ = a.split(), b.split()
        toks = []
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=A_, b=B_, autojunk=False).get_opcodes():
            if op == "equal":
                toks += A_[i1:i2]
            elif op == "delete":
                continue
            else:
                src = A_[i1:i2] if i2 > i1 else B_[j1:j2]
                k = key(src) or key(B_[j1:j2])
                code = codes.setdefault(k, f"X{len(codes) + 1}")
                last = (A_[i2 - 1] if i2 > i1 else B_[j2 - 1])
                poss = "'s" if re.search(r"(’s|'s)\W*$", B_[j2 - 1] if j2 > j1 else last) else ""
                tail = re.search(r"[.,;:!?)\"”]*$", B_[j2 - 1] if j2 > j1 else last).group(0)
                lead = re.match(r"^[(\"“]*", B_[j1] if j2 > j1 else A_[i1]).group(0)
                toks.append(f"{lead}{code}{poss}{tail}")
                n_spans += 1
        new_sents.append(re.sub(r"\s+", " ", " ".join(toks)).strip())
    return new_sents, n_spans


def neutralise_cjk(original, rewrite):
    """neutralise for Chinese (no spaces): a character-level diff; every changed span becomes an X-code."""
    codes, out, n = {}, [], 0
    for a, b in zip(original, rewrite):
        parts = []
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=list(a), b=list(b), autojunk=False).get_opcodes():
            if op == "equal":
                parts.append(a[i1:i2])
            elif op == "delete":
                continue
            else:
                k = a[i1:i2] or b[j1:j2]
                parts.append(" " + codes.setdefault(k, f"X{len(codes) + 1}") + " ")
                n += 1
        out.append(re.sub(r"\s+", " ", "".join(parts)).strip())
    return out, n


def realised_replacement(original, rewrite, banned_types):
    """drift_protocol.md check D2: share of the original's banned-type tokens gone from the rewrite (derivational
    rule), the share of all original content types that survive, and the banned types left."""
    orig_tokens = [stem(t) for s in original for t in tokens(s)] + ["#" + n for s in original for n in NUMBER.findall(CODE.sub(" ", s))]
    B = set(banned_types)
    T = content_types(rewrite, digits=True)
    surv = {b for b in B if types_banned(T, {b})}
    banned_tok = [t for t in orig_tokens if t in B]
    gone = [t for t in banned_tok if t not in surv]
    ot = content_types(original)
    return {"banned_token_share_removed": len(gone) / max(len(banned_tok), 1),
            "content_type_survival": len(ot & content_types(rewrite)) / max(len(ot), 1),
            "banned_types_left": sorted(surv)}


def drift_banned_types(work_id, sentences, level):
    """drift_protocol.md step 1: the nested set of content types (and numbers) to replace at a drift level (percent).
    Types are ordered by sha256(work_id|type) and the first round(level% x n) are banned, so 25 c 50 c 75 c 100."""
    types = sorted(content_types(sentences, digits=True), key=lambda t: hashlib.sha256(f"{work_id}|{t}".encode()).hexdigest())
    k = int(round(len(types) * level / 100))
    return types[:k]


DRIFT_EPITHETS = ["caped crusader", "dark knight", "the bat", "world's greatest detective", "clown prince", "boy wonder", "star-crossed",
                  "the shrew", "the moor", "the dane", "the bard", "the scottish play", "merchant of venice", "prince of denmark", "thane",
                  "the tempest"]


def drift_checks(original, rewrite, banned_types):
    """drift_protocol.md script checks: D1 alignment (sentence count, X-codes kept per sentence, none added), D1b
    length (each sentence within 20% or 3 words of the original), D1c capitals and epithets (no new capitalised
    token outside sentence starts and codes; no frozen epithet), D2 words (no banned type, derivational rule)."""
    fails = []
    if len(original) != len(rewrite):
        fails.append("D1 sentence count")
    for k, (a, b) in enumerate(zip(original, rewrite)):
        if sorted(CODE.findall(a)) != sorted(CODE.findall(b)):
            fails.append(f"D1 codes differ in sentence {k}")
        na, nb = len(a.split()), len(b.split())
        if abs(na - nb) > max(0.2 * na, 3):
            fails.append(f"D1b sentence {k} has {nb} words against {na}")
        toks = b.split()
        caps = [t for i, t in enumerate(toks) if i > 0 and _core(t)[:1].isupper() and not CODE.fullmatch(_core(t)) and _core(t) != "I"
                and not toks[i - 1].endswith((".", "!", "?", ":"))]
        if caps:
            fails.append(f"D1c capitalised token(s) {caps[:3]} in sentence {k}")
        if any(e in b.lower() for e in DRIFT_EPITHETS):
            fails.append(f"D1c epithet in sentence {k}")
    left = types_banned(content_types(rewrite, digits=True), set(banned_types))
    if left:
        fails.append(f"D2 banned types left: {sorted(left)[:5]}")
    return {"valid": not fails, "failures": fails}


def introduced_types(original, rewrite):
    """Content types a rewrite introduced (present in the rewrite, absent from the original): the source-leak index."""
    return content_types(rewrite) - content_types(original)


def eightgram_overlap(target_sentences, candidate_sentences, n=8):
    """Indices of the target's sentences that share a word 8-gram (codes stripped, lower case) with the candidate."""
    def grams(s):
        t = tokens(s)
        return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}
    cg = set().union(*[grams(s) for s in candidate_sentences]) if candidate_sentences else set()
    return [i for i, s in enumerate(target_sentences) if grams(s) & cg]


_TITLE_JUNK = {"film", "movie", "novel", "serial", "series", "the", "and", "part", "chapter", "story", "adventures", "return"}


def recognised(answer, title, entity=None):
    """Recognition probe rule: the LLM's answer names the work when it contains a distinctive title word (>= 4
    letters, not a stop word, number or generic word) or a distinctive word of the documented entity."""
    if not answer or re.search(r"\bunknown\b|don't know|do not know|not sure", answer, re.I):
        return False
    a = set(tokens(answer))
    t = re.sub(r"\([^)]*\)", " ", title or "")
    words = {w for w in tokens(t) if len(w) >= 4 and w not in STOP and w not in _TITLE_JUNK}
    if entity:
        words |= {w for w in tokens(re.sub(r"\([^)]*\)", " ", entity)) if len(w) >= 4 and w not in STOP and w not in _TITLE_JUNK}
    return bool(words & a)


# ---------------------------------------------------------------- planted-story checks
def story_checks(story, plot, memes, siblings=None):
    """planted_story_protocol.md script checks for one story version. story = {"plot", "version", "sentences",
    "slots": {meme_id: [sentence indices]}}; plot = its planted_design.json row; memes = {id: meme}; siblings = the
    other versions of the same plot.
      Q1 frame: 18-30 sentences; one slot per planted meme, 1-3 sentences each, disjoint; every version has the same
         length, slot positions and non-slot sentences
      Q1b length: each slot fill has within 15% of the explicit fill's word count (and at most 4 words apart for short
         fills)
      Q2 words (ban_hits, derivational families included): no planted meme's listed term in a frame sentence; none
         anywhere in the banned, implicit, coded and control versions; each explicit slot has >= 1 of its meme's
         terms; each decoy slot >= 2
      Q3 names: no franchise name (FRANCHISE) in any sentence
    Returns {"valid": bool, "failures": [...]}."""
    fails = []
    S = story["sentences"]
    slots = story.get("slots", {})
    if not 18 <= len(S) <= 30:
        fails.append(f"Q1 length {len(S)}")
    if sorted(slots) != sorted(plot["memes"]):
        fails.append("Q1 slots do not match the planted memes")
    used = [i for v in slots.values() for i in v]
    if len(used) != len(set(used)) or any(not 1 <= len(v) <= 3 for v in slots.values()) or any(not 0 <= i < len(S) for i in used):
        fails.append("Q1 slot indices")
    frame = [i for i in range(len(S)) if i not in set(used)]
    expl = None
    for o in siblings or []:
        if len(o["sentences"]) != len(S) or o.get("slots") != slots or any(o["sentences"][i] != S[i] for i in frame):
            fails.append(f"Q1 frame differs from the {o['version']} version")
        if o["version"] == "explicit":
            expl = o
    if expl is not None and story["version"] != "explicit" and not fails:
        for m, idx in slots.items():
            a, b = n_words([S[i] for i in idx]), n_words([expl["sentences"][i] for i in idx])
            if abs(a - b) > max(0.15 * b, 4):
                fails.append(f"Q1b slot {m} has {a} words against {b} in the explicit fill")
    terms = {m: memes[m]["keywords"] for m in plot["memes"]}
    allterms = [t for v in terms.values() for t in v]
    for i in frame:
        if ban_hits(S[i], allterms):
            fails.append(f"Q2 listed term in frame sentence {i}")
    v = story["version"]
    if v in ("banned", "implicit", "coded", "control"):
        for i in used:
            if ban_hits(S[i], allterms):
                fails.append(f"Q2 listed term in {v} slot sentence {i}")
    for m, idx in slots.items():
        k = sum(ban_hits(S[i], terms[m]) for i in idx)
        if v == "explicit" and k < 1:
            fails.append(f"Q2 explicit slot for {m} has no listed term")
        if v == "decoy" and k < 2:
            fails.append(f"Q2 decoy slot for {m} has fewer than 2 listed terms")
    for i, s in enumerate(S):
        if FRANCHISE.search(s):
            fails.append(f"Q3 franchise name in sentence {i}")
    return {"valid": not fails, "failures": fails}


def q4_q5_exclusions(version, planted, all_memes, claude_reads, other_reads=()):
    """Q4/Q5 (planted_story_protocol.md), symmetric. claude_reads: the memes each of the two Claude readers found;
    other_reads: the cross-family reader's. A planted meme passes in explicit, banned and coded versions when both
    Claude readers name it, in implicit versions when at least one does. A story is a negative for a meme (any
    unplanted meme, or a planted one in its control or decoy version) only when NO reader of any family names it;
    otherwise it is excluded for that meme. Returns the set of excluded memes."""
    excl = set()
    reads = list(claude_reads) + list(other_reads)
    for m in all_memes:
        hit_c = sum(m in f for f in claude_reads)
        hit_any = sum(m in f for f in reads)
        if m in planted and version in ("explicit", "banned", "coded"):
            if hit_c < 2:
                excl.add(m)
        elif m in planted and version == "implicit":
            if hit_c < 1:
                excl.add(m)
        elif hit_any >= 1:
            excl.add(m)
    return excl


def zh_checks(original, translation, ban_terms_zh=()):
    """translation_protocol.md L1 and L3: same sentence count, codes preserved, CJK in every sentence, no Latin word
    except codes, no franchise rendering (FRANCHISE_ZH), none of the Chinese ban-list terms."""
    fails = []
    if len(original) != len(translation):
        fails.append("L1 sentence count")
    for k, (a, b) in enumerate(zip(original, translation)):
        if sorted(CODE.findall(a)) != sorted(CODE.findall(b)):
            fails.append(f"L1 codes differ in sentence {k}")
        if not is_cjk(b):
            fails.append(f"L1 no CJK in sentence {k}")
        if re.search(r"[A-Za-z]{2,}", CODE.sub(" ", b)):
            fails.append(f"L1 Latin word in sentence {k}")
        if FRANCHISE_ZH.search(b):
            fails.append(f"L3 franchise rendering in sentence {k}")
        if any(t and t in b for t in ban_terms_zh):
            fails.append(f"L3 Chinese ban-list term in sentence {k}")
    return {"valid": not fails, "failures": fails}


# =============================================================================================== statistics
def auc(pos, neg):
    """Mann-Whitney AUC with ties counted as one half."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = sps.rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def signflip(d, clusters=None, alternative="greater", null_value=0.0, n_mc=N_SIGNFLIP_MC, rng=None):
    """Paired sign-flip permutation test of the mean of d (minus null_value). With clusters, d is first averaged
    within each cluster and clusters are flipped as units (each cluster weighs the same). Exact when 2**k <=
    EXACT_SIGNFLIP_MAX, Monte Carlo otherwise. p counts the observed pattern."""
    d = np.asarray(d, float)
    if clusters is not None:
        clusters = np.asarray(clusters)
        keys = list(dict.fromkeys(clusters.tolist()))
        d = np.array([np.nanmean(d[clusters == k]) if np.isfinite(d[clusters == k]).any() else np.nan for k in keys])
    d = d[np.isfinite(d)] - null_value
    k = len(d)
    if k == 0:
        return {"stat": float("nan"), "p": float("nan"), "n": 0, "exact": True}
    obs = d.mean()
    if 2 ** k <= EXACT_SIGNFLIP_MAX:
        null = np.zeros(2 ** k)
        for start in range(0, 2 ** k, 2 ** 16):
            idx = np.arange(start, min(start + 2 ** 16, 2 ** k))
            signs = 1 - 2 * ((idx[:, None] >> np.arange(k)) & 1)
            null[idx] = (signs * d).mean(1)
        exact = True
    else:
        rng = rng or rng_for("signflip", k)
        null = np.empty(n_mc)
        for start in range(0, n_mc, 20000):
            m = min(20000, n_mc - start)
            signs = rng.choice([-1.0, 1.0], size=(m, k))
            null[start:start + m] = (signs * d).mean(1)
        null = np.concatenate([[obs], null])
        exact = False
    eps = 1e-12
    if alternative == "greater":
        p = float(np.mean(null >= obs - eps))
    elif alternative == "less":
        p = float(np.mean(null <= obs + eps))
    else:
        p = float(np.mean(np.abs(null) >= abs(obs) - eps))
    return {"stat": float(obs + null_value), "p": p, "n": int(k), "exact": exact}


def cluster_means(d, clusters=None):
    d = np.asarray(d, float)
    if clusters is None:
        return d[np.isfinite(d)]
    clusters = np.asarray(clusters)
    keys = list(dict.fromkeys(clusters.tolist()))
    cm = np.array([np.nanmean(d[clusters == k]) if np.isfinite(d[clusters == k]).any() else np.nan for k in keys])
    return cm[np.isfinite(cm)]


def mean_ci(d, clusters=None, n=N_BOOT, key="ci"):
    """Bootstrap over clusters (or items) of the mean: the 95% two-sided interval and the one-sided 95% bounds."""
    cm = cluster_means(d, clusters)
    if len(cm) == 0:
        return {"ci95": [float("nan")] * 2, "lower95_one_sided": float("nan"), "upper95_one_sided": float("nan")}
    rng = rng_for("meanci", key, len(cm))
    bs = cm[rng.integers(0, len(cm), (n, len(cm)))].mean(1)
    return {"ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "lower95_one_sided": float(np.percentile(bs, 5)), "upper95_one_sided": float(np.percentile(bs, 95))}


def holm(pvals, alpha=ALPHA):
    """Holm-Bonferroni: {name: p} -> {name: {"p", "p_holm", "reject", "threshold"}}."""
    names = sorted(pvals, key=lambda k: (np.inf if not np.isfinite(pvals[k]) else pvals[k]))
    m = len(names)
    out, running, stop = {}, 0.0, False
    for i, k in enumerate(names):
        p = pvals[k] if np.isfinite(pvals[k]) else 1.0
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        rej = (not stop) and p <= alpha / (m - i)
        if not rej:
            stop = True
        out[k] = {"p": pvals[k], "p_holm": adj, "reject": bool(rej), "threshold": alpha / (m - i)}
    return out


def bh(pvals):
    """Benjamini-Hochberg q-values: {name: p} -> {name: q}."""
    names = [k for k in pvals if pvals[k] is not None and np.isfinite(pvals[k])]
    if not names:
        return {}
    p = np.array([pvals[k] for k in names])
    order = np.argsort(p)
    m = len(p)
    q = np.empty(m)
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, p[i] * m / rank)
        q[i] = prev
    return {k: float(v) for k, v in zip(names, q)}


def gini(x):
    x = np.sort(np.asarray(x, float))
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def cohen_kappa(a, b):
    """Cohen's kappa for two equal-length label sequences (any hashable labels)."""
    a, b = list(a), list(b)
    if not a:
        return float("nan")
    labs = sorted(set(a) | set(b), key=str)
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in labs) / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def mcnemar_exact(a_correct, b_correct, alternative="greater"):
    """Exact McNemar test on paired binary outcomes: is A right more often than B on the same items?"""
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return {"a_only": n10, "b_only": n01, "p": 1.0}
    alt = {"greater": "greater", "less": "less", "two-sided": "two-sided"}[alternative]
    return {"a_only": n10, "b_only": n01, "p": float(sps.binomtest(n10, n, 0.5, alternative=alt).pvalue)}


def iut(pvals):
    """Intersection-union test: a claim that must hold against every rival is tested at alpha by the largest p."""
    v = [p for p in pvals if p is not None]
    return float(max(v)) if v and all(np.isfinite(v)) else float("nan")


def falsification_test(d, sesoi, clusters=None, null_value=None):
    """One-sided test that the effect is smaller than the smallest effect of interest in the predicted direction
    (H0: mean(d) >= sesoi). Rejection at alpha (unadjusted) makes the hypothesis 'falsified'."""
    return signflip(d, clusters=clusters, alternative="less", null_value=sesoi if null_value is None else null_value)


def verdict3(supported, falsified, inconclusive=False, note=None):
    """Three-way verdict (PREREGISTRATION.md section 5): supported when the Holm test rejects; falsified when the
    one-sided 95% test excludes the smallest effect of interest; undetermined otherwise; inconclusive when a section-7
    check failed."""
    if inconclusive:
        return "inconclusive"
    if supported:
        v = "supported"
    elif falsified:
        v = "falsified"
    else:
        v = "undetermined"
    return v if note is None else f"{v} ({note})"


# =============================================================================================== representations
def cosine(A, B=None):
    A = np.asarray(A, np.float64)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    if B is None:
        return A @ A.T
    B = np.asarray(B, np.float64)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T


def lexical_text(sentences):
    return " ".join(NUMBER.sub(" ", CODE.sub(" ", s)) for s in sentences)


def tfidf_matrix(texts, fit_texts=None, ngram=(1, 1), min_df=2, sublinear=True):
    """TF-IDF (default: phylo.py settings, English stop words, min_df 2, sublinear tf; digits and X-codes removed for
    every lexical method). Chinese text (any CJK character) uses character uni- and bigrams."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    fit_texts = texts if fit_texts is None else fit_texts
    if any(is_cjk(t) for t in fit_texts):
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2), min_df=min_df, sublinear_tf=sublinear)
    else:
        vec = TfidfVectorizer(stop_words="english", ngram_range=ngram, min_df=min_df, sublinear_tf=sublinear)
    vec.fit(fit_texts)
    return vec.transform(texts), vec


def word_presence(texts, fit_texts=None):
    X, vec = tfidf_matrix(texts, fit_texts)
    return (X > 0).toarray(), vec


def lsa_vectors(X_fit, X, k=100, seed=0):
    from sklearn.decomposition import TruncatedSVD
    k = max(1, min(k, X_fit.shape[1] - 1, X_fit.shape[0] - 1))
    svd = TruncatedSVD(k, random_state=seed).fit(X_fit)
    return svd.transform(X)


def lda_fit_transform(fit_texts, texts, K, seed=0, max_iter=20):
    """LDA (sklearn, batch variational Bayes, default priors 1/K) on word counts with English stop words and
    min_df 2; returns doc-topic distributions for texts and the fitted (vectorizer, model)."""
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    if any(is_cjk(t) for t in fit_texts):
        cv = CountVectorizer(analyzer="char_wb", ngram_range=(1, 2), min_df=2)
    else:
        cv = CountVectorizer(stop_words="english", min_df=2)
    C = cv.fit_transform(fit_texts)
    lda = LatentDirichletAllocation(n_components=K, learning_method="batch", max_iter=max_iter, random_state=seed)
    lda.fit(C)
    return lda.transform(cv.transform(texts)), (cv, lda)


def reduce_for_clustering(E, seed=0):
    """BERTopic defaults: UMAP to 5 dimensions (15 neighbours, min_dist 0, cosine). If umap-learn is not installed the
    pre-registered fallback is PCA to 10 dimensions (a reported deviation if used)."""
    try:
        from umap import UMAP
        return UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed).fit_transform(E), "umap"
    except Exception:
        from sklearn.decomposition import PCA
        return PCA(min(10, E.shape[1]), random_state=seed).fit_transform(E), "pca10"


def cluster_topics(E, min_cluster_size=15, seed=0):
    """BERTopic-style topics without the bertopic package: reduce, then HDBSCAN (sklearn). Returns labels (-1 =
    noise), centroids in the reduced space, the reduced points, the reducer and the membership probabilities."""
    from sklearn.cluster import HDBSCAN
    R, how = reduce_for_clustering(np.asarray(E, np.float32), seed)
    hd = HDBSCAN(min_cluster_size=min_cluster_size, copy=True).fit(R)
    lab = hd.labels_
    ks = sorted(set(lab) - {-1})
    cents = np.array([R[lab == k].mean(0) for k in ks]) if ks else np.zeros((0, R.shape[1]))
    return lab, cents, R, how, np.asarray(getattr(hd, "probabilities_", np.ones(len(lab))))


def ctfidf_top_words(sentences, labels, n=10):
    """c-TF-IDF (BERTopic): words that characterise each cluster; used to label topics for T6."""
    from sklearn.feature_extraction.text import CountVectorizer
    ks = sorted(set(labels) - {-1})
    docs = [" ".join(sentences[i] for i in np.where(labels == k)[0]) for k in ks]
    cv = CountVectorizer(stop_words="english", min_df=1)
    C = cv.fit_transform(docs).toarray().astype(float)
    tf = C / (C.sum(1, keepdims=True) + 1e-12)
    A_ = C.sum() / max(len(ks), 1)
    idf = np.log(1 + A_ / (C.sum(0) + 1e-12))
    W = tf * idf
    vocab = np.array(cv.get_feature_names_out())
    return {int(k): vocab[np.argsort(-W[i])[:n]].tolist() for i, k in enumerate(ks)}


def umass_coherence(topic_word, doc_word_bin, n_top=10):
    """UMass coherence of each LDA topic over binary document-word occurrence (T6 best-of arm)."""
    D = np.asarray(doc_word_bin.toarray() if hasattr(doc_word_bin, "toarray") else doc_word_bin, bool)
    out = []
    for row in topic_word:
        top = np.argsort(-row)[:n_top]
        s = 0.0
        for i in range(1, len(top)):
            for j in range(i):
                co = np.sum(D[:, top[i]] & D[:, top[j]])
                s += math.log((co + 1) / max(D[:, top[j]].sum(), 1))
        out.append(s)
    return np.array(out)


def topk_mean(values, k=TOPK):
    v = np.asarray(values, np.float64)
    k = min(k, v.shape[0])
    return np.partition(v, -k, axis=0)[-k:].mean(0)


def work_from_sentences(sent_scores, sent_work, n_works, how="top3"):
    """Aggregate sentence scores [n_sent, ...] to works: 'top3' = mean of the 3 strongest (every sentence-level method,
    as the neurons do); 'mean' = plain mean."""
    sent_work = np.asarray(sent_work)
    sent_scores = np.asarray(sent_scores)
    out = np.zeros((n_works,) + sent_scores.shape[1:])
    order = np.argsort(sent_work, kind="stable")
    bounds = np.searchsorted(sent_work[order], np.arange(n_works + 1))
    for w in range(n_works):
        rows = order[bounds[w]:bounds[w + 1]]
        if len(rows) == 0:
            out[w] = np.nan
        elif how == "top3":
            out[w] = topk_mean(sent_scores[rows])
        else:
            out[w] = np.asarray(sent_scores[rows], np.float64).mean(0)
    return out


# ---------------------------------------------------------------- linear detectors (vectors, hidden states, lexical probe)
PROBE_C = 1.0


def _norm_rows(X):
    X = np.asarray(X, np.float64)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def fit_linear(kind, P, B, C=PROBE_C):
    """A detector from positive rows P and background rows B (already scaled: embeddings L2-normalised, hidden states
    reference-standardised). 'cproto': the contrastive prototype direction mean(P) - mean(B), scored by cosine;
    'probe': L2 logistic regression (fixed C = 1, balanced classes), scored by its decision function."""
    P, B = np.asarray(P, np.float64), np.asarray(B, np.float64)
    if kind == "cproto":
        w = P.mean(0) - B.mean(0)
        return {"kind": "cproto", "w": (w / (np.linalg.norm(w) + 1e-12)).tolist(), "b": 0.0}
    from sklearn.linear_model import LogisticRegression
    X = np.vstack([P, B])
    y = np.r_[np.ones(len(P)), np.zeros(len(B))]
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X, y)
    return {"kind": "probe", "w": lr.coef_[0].tolist(), "b": float(lr.intercept_[0])}


def score_linear(X, det):
    X = np.asarray(X, np.float64)
    w = np.asarray(det["w"])
    if det["kind"] == "cproto":
        return _norm_rows(X) @ w
    return X @ w + det["b"]


def _pct_of(x, bg):
    bg = np.sort(np.asarray(bg, float))
    l, r = np.searchsorted(bg, x, "left"), np.searchsorted(bg, x, "right")
    return (l + 0.5 * (r - l)) / len(bg)


def loeo_linear(kind, P, B, C=PROBE_C):
    """Leave-one-example-out criterion: for each positive, fit on the others and take the held-out positive's
    percentile among the background scores; the mean over positives."""
    P = np.asarray(P, np.float64)
    vals = []
    for i in range(len(P)):
        det = fit_linear(kind, np.delete(P, i, 0), B, C)
        vals.append(float(_pct_of(score_linear(P[i:i + 1], det), score_linear(B, det))[0]))
    return float(np.mean(vals))


def fit_keyword_logreg(pos_sents, bg_sents, vec=None, C=PROBE_C):
    """keywords:logreg: L2 logistic regression on TF-IDF word uni- and bigrams of the positive sentences against the
    contrastive background. vec: a shared TfidfVectorizer fitted on all DEV sentences (fit_lexical_vectorizer)."""
    from sklearn.linear_model import LogisticRegression
    X = vec.transform(list(pos_sents) + list(bg_sents))
    y = np.r_[np.ones(len(pos_sents)), np.zeros(len(bg_sents))]
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X, y)
    return {"kind": "logreg", "coef": lr.coef_[0].astype(np.float32).tolist(), "b": float(lr.intercept_[0])}


def fit_lexical_vectorizer(sentences):
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1).fit(
        [lexical_text([s]) for s in sentences])


def lexical_vectorizer_from(vocab, idf):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, vocabulary=vocab)
    vec.fit(["x"])
    vec.idf_ = np.asarray(idf)
    return vec


def score_keyword_logreg(sentences, det, vec):
    X = vec.transform([lexical_text([s]) for s in sentences])
    return np.asarray(X @ np.asarray(det["coef"], np.float64)).ravel() + det["b"]


def wordnet_expand(terms, per_term=5):
    """keywords:expanded and keywords:tfidf-wordnet: each one-word term plus up to 5 single-word WordNet synonyms
    (lemmas of its first 3 noun and verb synsets, in WordNet order). Without nltk's WordNet the terms are returned
    unchanged and the result says so (a reported deviation)."""
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("test")
    except Exception:
        return list(terms), False
    out = list(terms)
    seen = {t.lower() for t in terms}
    for t in terms:
        if len(t.split()) != 1:
            continue
        added = 0
        for pos in ("n", "v"):
            for syn in wn.synsets(t, pos=pos)[:3]:
                for lem in syn.lemma_names():
                    w = lem.replace("_", " ").lower()
                    if " " in w or w in seen or not w.isalpha():
                        continue
                    out.append(w)
                    seen.add(w)
                    added += 1
                    if added >= per_term:
                        break
                if added >= per_term:
                    break
    return out, True


def tfidf_keywords(examples, background, n=10):
    """Automatic keyword query ('keywords:tfidf'): unigrams and bigrams, English stop words, sublinear tf, fitted on
    background + examples; term score = mean weight in the examples minus mean weight in the background; keep terms
    with a positive score that occur in at least 2 examples; top n."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    X = vec.fit_transform(list(background) + list(examples))
    Xb, Xe = X[:len(background)], X[len(background):]
    sc = np.asarray(Xe.mean(0)).ravel() - np.asarray(Xb.mean(0)).ravel()
    df_ex = np.asarray((Xe > 0).sum(0)).ravel()
    vocab = np.array(vec.get_feature_names_out())
    ok = (sc > 0) & (df_ex >= 2)
    order = [i for i in np.argsort(-sc) if ok[i]]
    return vocab[order[:n]].tolist()


def ban_list_overlap(query, ban):
    """Share of a query's terms that fall on the meme's ban list (stem overlap with any ban-list term's stems)."""
    bs = {s for t in ban for s in term_stems(t) if s not in STOP}
    if not query:
        return float("nan")
    return float(np.mean([bool(set(s for s in term_stems(q) if s not in STOP) & bs) for q in query]))


# =============================================================================================== neurons
def presence_bar(S_ref, rank=2):
    """Per-unit presence bar: the rank-th highest score among the reference films (rank 2: a unit is present in a work
    when its score beats all but one of them; rank 1, the strict T4 bar: it beats all of them)."""
    S_ref = np.asarray(S_ref, np.float32)
    return np.partition(S_ref, -rank, axis=0)[-rank]


def presence(S, S_ref, rank=2):
    return np.asarray(S, np.float32) > presence_bar(S_ref, rank)[None, :]


def optional_band(P_lineage, lo=2, hi=None, frac=0.5):
    """Units present in between lo and hi works of the lineage (hi default floor(frac x n); E2's band is 2-16 for 32)."""
    n = P_lineage.shape[0]
    hi = int(n * frac) if hi is None else hi
    c = P_lineage.sum(0)
    return (c >= lo) & (c <= hi)


def containment(P, rows, cand_rows, opt, weights=None):
    """R[i, j] = (weighted) share of candidate j's optional units that are also present in row i (E2's inheritance
    rate; weights = per-unit IDF weights for the '@idf' configuration)."""
    P = np.asarray(P, bool)
    w = np.ones(P.shape[1]) if weights is None else np.asarray(weights, float)
    R = np.zeros((len(rows), len(cand_rows)))
    Pr = P[list(rows)].astype(np.float64)
    for j, c in enumerate(cand_rows):
        oc = (opt & P[c]).astype(np.float64) * w
        denom = max(oc.sum(), 1e-12)
        R[:, j] = (Pr @ oc) / denom
    return R


def unit_idf(P_lineage):
    n = P_lineage.shape[0]
    return np.log(n / np.maximum(P_lineage.sum(0), 1))


def unit_scale(S_ref):
    S_ref = np.asarray(S_ref, np.float32)
    return np.median(S_ref, 0), S_ref.std(0) + 1e-6


def _colwise_searchsorted(sorted_bg, x, side, flat=None):
    """Vectorised per-column searchsorted: sorted_bg [n, U] sorted along axis 0, x [m, U] -> counts [m, U]."""
    n, U = sorted_bg.shape
    if flat is None:
        flat = _flat_sorted(sorted_bg, x)
    lo, span = flat[1], flat[2]
    off = np.arange(U, dtype=np.float64) * span
    q = (np.clip(x.astype(np.float64), lo, lo + span - 1.0) - lo + off[None, :])
    pos = np.searchsorted(flat[0], q.ravel(), side=side).reshape(q.shape)
    return pos - (np.arange(U) * n)[None, :]


def _flat_sorted(sorted_bg, x=None):
    lo = float(sorted_bg.min()) - 1.0
    hi = float(sorted_bg.max()) + 1.0
    if x is not None:
        lo, hi = min(lo, float(x.min()) - 1.0), max(hi, float(x.max()) + 1.0)
    span = hi - lo + 1.0
    U = sorted_bg.shape[1]
    off = np.arange(U, dtype=np.float64) * span
    return (sorted_bg.astype(np.float64) - lo + off[None, :]).T.ravel(), lo, span


def select_units(examples, background, m=1, chunk=50000, exclude=None):
    """Few-shot unit selection: score every unit by the mean over examples of the example's mid-rank percentile within
    the unit's background distribution; ties broken by the mean margin (x - q99) / (q99 - q50). Returns the m best
    units (best first), their scores and margins."""
    examples = np.asarray(examples, np.float32)
    U = examples.shape[1]
    score, margin = np.zeros(U), np.zeros(U)
    for c0 in range(0, U, chunk):
        bg = np.sort(np.asarray(background[:, c0:c0 + chunk], np.float32), axis=0)
        ex = examples[:, c0:c0 + chunk]
        n = bg.shape[0]
        flat = _flat_sorted(bg, ex)
        left = _colwise_searchsorted(bg, ex, "left", flat)
        right = _colwise_searchsorted(bg, ex, "right", flat)
        score[c0:c0 + chunk] = ((left + 0.5 * (right - left)) / n).mean(0)
        q50, q99 = bg[int(0.50 * (n - 1))], bg[int(0.99 * (n - 1))]
        margin[c0:c0 + chunk] = ((ex - q99[None]) / (q99 - q50 + 1e-6)[None]).mean(0)
    if exclude is not None:
        score[np.asarray(exclude)] = -1
    order = np.lexsort((-margin, -score))
    best = order[:m]
    return best.tolist(), score[best].tolist(), margin[best].tolist()


class RowCat:
    """Row-concatenation of several [n_i, U] arrays (memory-mapped or not): indexing [:, cols] stacks the columns."""

    def __init__(self, parts):
        self.parts = parts
        self.shape = (sum(p.shape[0] for p in parts), parts[0].shape[1])

    def __getitem__(self, key):
        rows, cols = key
        X = np.concatenate([np.asarray(p[:, cols], np.float32) for p in self.parts])
        return X if rows == slice(None) else X[rows]


def background_standardiser(background, units, drop=None):
    """(x - q50) / (q99 - q50 + 1e-6) per selected unit, from the background rows (minus drop)."""
    bg = np.asarray(background[:, list(units)], np.float32)
    if drop is not None and len(drop):
        bg = np.delete(bg, np.asarray(drop), 0)
    bg = np.sort(bg, axis=0)
    n = bg.shape[0]
    q50, q99 = bg[int(0.50 * (n - 1))], bg[int(0.99 * (n - 1))]
    return q50, (q99 - q50 + 1e-6)


def neuron_detectors(B, pos_rows, own_rows, chunk=25000, n_top=5, folds=True, timer=None):
    """One pass over all units that fits every T2 neuron detector and its leave-one-example-out criterion.

    B: [n, U] rows = the background sentences followed by every example sentence of every meme (all one sentence per
    document after the same prefix). pos_rows[(meme, set)]: rows of B that are the set's positives; own_rows[meme]:
    every positive row of the meme (excluded from that meme's background, so its background is the reference
    sentences, the other memes' sentences and its own hard negatives). Returns {(meme, set): {"m1": unit, "m5":
    [units], "m1_loeo_units": [...], "m5_loeo_units": [[...]], "score", "margin"}}; held-out percentiles are
    computed afterwards (neuron_loeo_criteria)."""
    n, U = B.shape
    keys = list(pos_rows)
    best = {k: [] for k in keys}                 # full fit: list of (score, margin, unit), top n_top
    fold_best = {k: [[] for _ in pos_rows[k]] for k in keys}
    for c0 in range(0, U, chunk):
        cols = slice(c0, min(c0 + chunk, U))
        Xc = B[:, cols]
        Sc = np.sort(Xc, axis=0)
        flat = _flat_sorted(Sc, Xc)
        q50, q99 = Sc[int(0.50 * (n - 1))], Sc[int(0.99 * (n - 1))]
        scale = (q99 - q50 + 1e-6)
        pct_cache = {}
        for k in keys:
            m, st = k
            rows = pos_rows[k]
            P = Xc[rows]
            R = Xc[own_rows[m]]
            left = _colwise_searchsorted(Sc, P, "left", flat) - (R[None] < P[:, None]).sum(1)
            right = _colwise_searchsorted(Sc, P, "right", flat) - (R[None] <= P[:, None]).sum(1)
            pct = (left + 0.5 * (right - left)) / (n - len(own_rows[m]))
            mg = (P - q99[None]) / scale[None]
            pct_cache[k] = pct
            sc, mm = pct.mean(0), mg.mean(0)
            top = np.lexsort((-mm, -sc))[:n_top]
            best[k] = sorted(best[k] + [(float(sc[i]), float(mm[i]), int(c0 + i)) for i in top], key=lambda x: (-x[0], -x[1]))[:n_top]
            if folds and len(rows) > 1:
                ssum, msum = pct.sum(0), mg.sum(0)
                for i in range(len(rows)):
                    sci = (ssum - pct[i]) / (len(rows) - 1)
                    mmi = (msum - mg[i]) / (len(rows) - 1)
                    topi = np.lexsort((-mmi, -sci))[:n_top]
                    fold_best[k][i] = sorted(fold_best[k][i] + [(float(sci[j]), float(mmi[j]), int(c0 + j), float(pct[i, j])) for j in topi],
                                             key=lambda x: (-x[0], -x[1]))[:n_top]
    out = {}
    for k in keys:
        out[k] = {"m1": best[k][0][2], "m5": [u for _, _, u in best[k]], "score": [s for s, _, _ in best[k]],
                  "margin": [g for _, g, _ in best[k]]}
        if folds:
            out[k]["m1_loeo_units"] = [fb[0][2] for fb in fold_best[k]]
            out[k]["m1_loeo_heldout_pct"] = [fb[0][3] for fb in fold_best[k]]
            out[k]["m5_loeo_units"] = [[u for _, _, u, _ in fb] for fb in fold_best[k]]
    return out


def neuron_loeo_criteria(B, pos_rows, own_rows, det):
    """Leave-one-example-out criterion for the m1 and m5 neuron variants: the held-out positive's percentile among
    the meme's background (for m5, on the mean of the 5 background-standardised units)."""
    out = {}
    n = B.shape[0]
    for k, d in det.items():
        m, _ = k
        rows = pos_rows[k]
        if "m1_loeo_units" not in d:
            continue
        keep = np.setdiff1d(np.arange(n), own_rows[m])
        m1 = float(np.mean(d["m1_loeo_heldout_pct"]))
        vals = []
        for i, us in enumerate(d["m5_loeo_units"]):
            X = B[:, us]
            q50, sc = background_standardiser(B, us, drop=own_rows[m])
            z = ((X - q50) / sc).mean(1)
            vals.append(float(_pct_of(z[[rows[i]]], z[keep])[0]))
        out[k] = {"m1": m1, "m5": float(np.mean(vals))}
    return out


# =============================================================================================== lineage
def hub_groups_fixed(n_rows, hub_rows, n_cands):
    """Hub rule for Batman (E2): every candidate column is z-scored against the same fixed hub rows."""
    H = np.zeros((n_rows, 1), bool)
    H[list(hub_rows), 0] = True
    return H, np.zeros(n_cands, int), False


def hub_groups_other_targets(n_rows, target_rows, sources, n_cands):
    """Hub rule when every row adapts some candidate (Shakespeare): column c is z-scored against every row whose
    documented sources do not include c. The rule depends on the source labels, so the null recomputes it."""
    H = np.zeros((n_rows, n_cands), bool)
    for r, src in zip(target_rows, sources):
        H[r, :] = True
        H[r, list(src)] = False
    return H, np.arange(n_cands), True


def _zscores(Sim, target_rows, H, g, floor_q=HUB_SD_FLOOR_Q):
    """z[t, c] = (Sim[t, c] - mean over hub rows of group g[c], excluding t) / SD, the SD floored at the floor_q
    quantile of all positive SDs (so a near-constant column cannot explode)."""
    Sim = np.asarray(Sim, np.float64)
    S1 = H.T.astype(np.float64) @ Sim
    S2 = H.T.astype(np.float64) @ (Sim ** 2)
    cnt = H.sum(0).astype(np.float64)
    C = Sim.shape[1]
    cols = np.arange(C)
    X = Sim[target_rows]
    m = H[np.asarray(target_rows)][:, g].astype(np.float64)
    n = cnt[g][None, :] - m
    s1 = S1[g, cols][None, :] - m * X
    s2 = S2[g, cols][None, :] - m * X ** 2
    mean = s1 / np.maximum(n, 1)
    sd = np.sqrt(np.maximum(s2 / np.maximum(n, 1) - mean ** 2, 0))
    pos = sd[sd > 1e-12]
    floor = float(np.quantile(pos, floor_q)) if pos.size else 1e-9
    return (X - mean) / np.maximum(sd, floor)


def _shares_batch(Z, src_mask, eligible, ties="half"):
    """Shares for one or many relabellings. Z [T, C]; src_mask [b, T, C] (or [T, C]) marks each target's (mapped)
    documented sources. Share = (# eligible non-source candidates whose z beats the best source + one half of the
    ties) / # eligible non-source candidates. ties='strict' (E2's published rule) counts ties as not beating."""
    single = src_mask.ndim == 2
    S = src_mask[None] if single else src_mask
    Zb = np.broadcast_to(Z, S.shape)
    best = np.where(S, Zb, -np.inf).max(2)
    others = eligible[None] & ~S
    gt = ((Zb > best[..., None] + 1e-12) & others).sum(2)
    eq = ((np.abs(Zb - best[..., None]) <= 1e-12) & others).sum(2)
    n = np.maximum(others.sum(2), 1)
    sh = (gt + 0.5 * eq) / n if ties == "half" else gt / n
    sh = np.where(others.sum(2) == 0, 0.0, sh)
    return sh[0] if single else sh


def _src_mask(sources, T, C, perm=None):
    M = np.zeros((T, C), bool)
    for t, ss in enumerate(sources):
        M[t, [perm[s] if perm is not None else s for s in ss]] = True
    return M


def time_consistent_perms(C, sources, eligible, n, rng, batch=4096, max_draws=20_000_000):
    """n random permutations of the C candidates such that every relabelled documented source is eligible for its
    target (published no later than it): rejection sampling, exact for the time-consistent null."""
    pairs = [(t, s) for t, ss in enumerate(sources) for s in ss]
    tt = np.array([t for t, _ in pairs])
    ss = np.array([s for _, s in pairs])
    if eligible.all():
        return np.array([rng.permutation(C) for _ in range(n)]), 1.0
    out, draws = [], 0
    while sum(len(o) for o in out) < n:
        P = rng.random((batch, C)).argsort(1)
        ok = eligible[tt[None, :], P[:, ss]].all(1)
        out.append(P[ok])
        draws += batch
        if draws > max_draws:
            raise RuntimeError("time-consistent permutations too rare")
    P = np.concatenate(out)[:n]
    return P, float(sum(len(o) for o in out) / draws)


def lineage_test(Sim, target_rows, sources, eligible, hub, weights=None, use_hub=True, n_perm=N_PERM_LINEAGE,
                 rng=None, keep_null=False, ties="half"):
    """The E2 statistic for any lineage and any method.

    Sim [n_rows, n_cands]: similarity of each row work (targets and hub works) to each candidate source. sources: per
    target, the documented source columns. eligible [n_targets, n_cands]: candidates that may be ranked (time rule).
    hub: from hub_groups_*. Statistic: weighted mean over targets of the share of eligible non-source candidates whose
    hub-adjusted similarity beats the best documented source (ties count one half; lower = better recovery). Null:
    joint relabelling of candidate identities restricted to time-consistent relabellings (every relabelled source is
    eligible for its target); the hub rule is recomputed when it depends on the labels. One-sided p = P(null <= obs)."""
    rng = rng or rng_for("lineage", np.asarray(Sim).shape)
    H, g, dep = hub
    eligible = np.asarray(eligible, bool)
    T, C = len(target_rows), np.asarray(Sim).shape[1]
    w = np.full(T, 1.0 / T) if weights is None else np.asarray(weights, float) / np.sum(weights)
    ident = np.arange(C)

    def z_for(gmap):
        if not use_hub:
            return np.asarray(Sim, np.float64)[target_rows]
        return _zscores(Sim, target_rows, H, gmap)

    Z0 = z_for(g)
    M0 = _src_mask(sources, T, C)
    sh = _shares_batch(Z0, M0, eligible, ties)
    best = np.where(M0, Z0, -np.inf).max(1)
    rk = np.array([1 + int(((Z0[t] > best[t] + 1e-12) & eligible[t] & ~M0[t]).sum()) for t in range(T)])
    md = np.array([bool(best[t] > np.median(Z0[t][eligible[t] & ~M0[t]])) if (eligible[t] & ~M0[t]).any() else True for t in range(T)])
    obs = float(w @ sh)
    perms, acc = time_consistent_perms(C, sources, eligible, n_perm, rng)
    null = np.empty(n_perm)
    per_t = np.zeros(T)
    if dep and use_hub:
        for i, pi in enumerate(perms):
            inv = np.empty(C, int)
            inv[pi] = ident
            s_i = _shares_batch(z_for(g[inv]), _src_mask(sources, T, C, pi), eligible, ties)
            null[i] = w @ s_i
            per_t += s_i
    else:
        for b0 in range(0, n_perm, 2000):
            Pb = perms[b0:b0 + 2000]
            masks = np.zeros((len(Pb), T, C), bool)
            for t, ss in enumerate(sources):
                for s in ss:
                    masks[np.arange(len(Pb)), t, Pb[:, s]] = True
            s_b = _shares_batch(Z0, masks, eligible, ties)
            null[b0:b0 + len(Pb)] = s_b @ w
            per_t += s_b.sum(0)
    out = {"stat": obs, "null_mean": float(null.mean()), "p": float((np.sum(null <= obs + 1e-12) + 1) / (n_perm + 1)),
           "per_target_share": sh.tolist(), "per_target_null_mean": (per_t / n_perm).tolist(), "per_target_rank": rk.tolist(),
           "beats_median": int(md.sum()), "n_targets": T, "perm_acceptance": acc}
    if keep_null:
        out["null"] = null
    return out


def pooled_lineage(results, weights):
    """Joint test over independent lineages (H4's conjunct): obs = sum_i w_i stat_i, null draws combined draw by draw
    from each lineage's own null (keep_null=True, the same n_perm)."""
    w = np.asarray(weights, float) / np.sum(weights)
    obs = float(sum(wi * r["stat"] for wi, r in zip(w, results)))
    null = sum(wi * np.asarray(r["null"]) for wi, r in zip(w, results))
    return {"stat": obs, "null_mean": float(null.mean()), "p": float((np.sum(null <= obs + 1e-12) + 1) / (len(null) + 1))}


def residual_sim(Sim_main, Sim_lex_list, fit_mask):
    """Residual of Sim_main after OLS on the lexical similarity matrices (plus intercept) over the fit_mask pairs."""
    y = np.asarray(Sim_main, float)
    X = np.column_stack([np.ones(fit_mask.sum())] + [np.asarray(x, float)[fit_mask] for x in Sim_lex_list])
    beta = np.linalg.lstsq(X, y[fit_mask], rcond=None)[0]
    Xall = np.column_stack([np.ones(y.size)] + [np.asarray(x, float).ravel() for x in Sim_lex_list])
    resid = y - (Xall @ beta).reshape(y.shape)
    r2 = 1 - np.var(resid[fit_mask]) / max(np.var(y[fit_mask]), 1e-12)
    return resid, float(r2)


def advantage_test(shares_neuron, shares_rival, clusters=None, sesoi=None, key="adv"):
    """Head-to-head on one lineage set: advantage d = share(rival) - share(neurons) per target (positive = neurons
    recover the documented source better), averaged within clusters, sign-flip over clusters (one-sided, greater), a
    cluster bootstrap interval and, with sesoi, the falsification test."""
    d = np.asarray(shares_rival, float) - np.asarray(shares_neuron, float)
    r = signflip(d, clusters=clusters, alternative="greater")
    r.update(mean_ci(d, clusters, key=key))
    if sesoi is not None:
        r["falsification"] = falsification_test(d, sesoi, clusters)
    r["per_target_d"] = d.tolist()
    return r


def method_contrast(shares_a, shares_b, clusters=None, alternative="less"):
    """Generic head-to-head: d = share_A - share_B per target (negative = A better), sign-flip over clusters."""
    d = np.asarray(shares_a, float) - np.asarray(shares_b, float)
    r = signflip(d, clusters=clusters, alternative=alternative)
    r.update(mean_ci(d, clusters, key="contrast"))
    return r


def h4_head_to_head(rival_by_level, neuron_by_level, clusters, levels=H4_LEVELS, sesoi=SESOI["H4"]):
    """H4: d(t) = mean over levels 50, 75, 100 of share_rival(t, L) - share_neurons(t, L) (positive = neurons
    recover better under heavy drift), sign-flip over clusters. The level-0 difference and every level's difference
    are reported beside it."""
    d = np.mean([np.asarray(rival_by_level[l], float) - np.asarray(neuron_by_level[l], float) for l in levels], 0)
    r = signflip(d, clusters=clusters, alternative="greater")
    r.update(mean_ci(d, clusters, key="h4"))
    r["falsification"] = falsification_test(d, sesoi, clusters)
    r["level0_advantage"] = float(cluster_means(np.asarray(rival_by_level[0]) - np.asarray(neuron_by_level[0]), clusters).mean())
    r["by_level_advantage"] = {int(l): float(cluster_means(np.asarray(rival_by_level[l]) - np.asarray(neuron_by_level[l]), clusters).mean())
                               for l in rival_by_level}
    r["per_target_d"] = d.tolist()
    return r


def retention_contrast(rival_by_level, neuron_by_level, rival_null, neuron_null, clusters, levels=H4_LEVELS, min_margin=0.02):
    """Secondary for H4: headroom-free retention. Per cluster, rho_x(L) = sum_t (null_t - share_t(L)) / sum_t
    (null_t - share_t(0)) with each target's permutation-null mean; rho averaged over levels 50-100; d = rho_neurons -
    rho_rival, sign-flip over clusters. A cluster whose level-0 margin is below min_margin for either method is left
    out (nothing to retain)."""
    cl = np.asarray(clusters)
    keys = list(dict.fromkeys(cl.tolist()))
    d = []
    for k in keys:
        rows = cl == k
        rr = []
        for by, nul in ((neuron_by_level, neuron_null), (rival_by_level, rival_null)):
            den = float(np.sum(np.asarray(nul[0])[rows] - np.asarray(by[0])[rows]))
            if den < min_margin * rows.sum():
                rr.append(np.nan)
                continue
            rr.append(np.mean([float(np.sum(np.asarray(nul[l])[rows] - np.asarray(by[l])[rows])) / den for l in levels]))
        d.append(rr[0] - rr[1])
    d = np.array(d)
    r = signflip(d, alternative="greater")
    r.update(mean_ci(d, key="retention"))
    r["n_clusters_used"] = int(np.isfinite(d).sum())
    return r


def drift_interaction(shares_by_level_a, shares_by_level_b, clusters=None):
    """Descriptive only (the draft's H4 statistic, kept for comparison): degradation D_x(t) = mean over levels > 0
    of share(t, L) - share(t, 0); d = D_a - D_b."""
    lv = [l for l in sorted(shares_by_level_a) if l > 0]
    Da = np.mean([np.asarray(shares_by_level_a[l]) - np.asarray(shares_by_level_a[0]) for l in lv], 0)
    Db = np.mean([np.asarray(shares_by_level_b[l]) - np.asarray(shares_by_level_b[0]) for l in lv], 0)
    cl = np.arange(len(Da)) if clusters is None else np.asarray(clusters)
    r = signflip(Da - Db, clusters=cl, alternative="greater")
    r["degradation_a"] = float(cluster_means(Da, cl).mean())
    r["degradation_b"] = float(cluster_means(Db, cl).mean())
    return r


# =============================================================================================== meme detection (T2)
def _usable(row, meme, itt=False):
    """Per protocol, a story is left out for a meme that quality control excluded it for (Q4-Q5) and for dropped
    memes; intention-to-treat keeps every valid story as designed."""
    if itt:
        return True
    return meme not in row.get("exclude_memes", ()) and meme not in row.get("dropped_memes", ())


def _plants(row, meme):
    return meme in row["memes"] and row["version"] in PLANTED


def twin_pairs(design, meme, condition, valid=None, itt=False, skip_plots=()):
    """Within-plot pairs for a meme: (positive story, its own plot's control story, plot) for every plot that plants
    the meme and every version in the condition. Positive and negative share the frame word for word."""
    valid = np.ones(len(design), bool) if valid is None else np.asarray(valid)
    idx = {(r["plot"], r["version"]): i for i, r in enumerate(design)}
    out = []
    for i, r in enumerate(design):
        if r["version"] in condition and meme in r["memes"] and r["plot"] not in skip_plots and valid[i] and _usable(r, meme, itt):
            j = idx.get((r["plot"], "control"))
            if j is not None and valid[j] and _usable(design[j], meme, itt):
                out.append((i, j, r["plot"]))
    return out


def _pair_ind(col, pairs):
    a = np.array([col[i] for i, _, _ in pairs], float)
    b = np.array([col[j] for _, j, _ in pairs], float)
    ok = np.isfinite(a) & np.isfinite(b)
    return np.where(a > b, 1.0, np.where(a == b, 0.5, 0.0))[ok], np.array([p for _, _, p in pairs])[ok]


def detection_aucs(scores, design, memes, condition, valid=None, scheme="twin", itt=False):
    """AUC per meme for one method (scores [n_stories, n_memes]).
      twin (primary): positives = the meme's planted versions in the condition; each is compared with its own plot's
        control version (same frame, equal-stakes control event); AUC = share of pairs won (ties one half)
      matched: negatives = the same-condition versions of plots that do not plant the meme (salience matched)
      all: negatives = every story not containing the meme (the draft's rule)
      hard: negatives = the meme's own plots' control and decoy versions
      decoy: the meme's decoy version against its own control (S11 construction check; keywords should 'win')
      vs_decoy: positives against their own plot's decoy version"""
    scores = np.asarray(scores, float)
    valid = np.ones(len(design), bool) if valid is None else np.asarray(valid)
    out = {}
    idx = {(r["plot"], r["version"]): i for i, r in enumerate(design)}
    for j, m in enumerate(memes):
        col = scores[:, j]
        if scheme == "twin":
            ind, _ = _pair_ind(col, twin_pairs(design, m, condition, valid, itt))
            out[m] = float(ind.mean()) if len(ind) else float("nan")
            continue
        if scheme in ("decoy", "vs_decoy"):
            prs = []
            for i, r in enumerate(design):
                if scheme == "decoy" and r["version"] == "decoy" and m in r["memes"]:
                    k = idx.get((r["plot"], "control"))
                elif scheme == "vs_decoy" and r["version"] in condition and m in r["memes"]:
                    k = idx.get((r["plot"], "decoy"))
                else:
                    continue
                if k is not None and valid[i] and valid[k] and _usable(r, m, itt) and _usable(design[k], m, itt):
                    prs.append((i, k, r["plot"]))
            ind, _ = _pair_ind(col, prs)
            out[m] = float(ind.mean()) if len(ind) else float("nan")
            continue
        ok = [bool(valid[i]) and _usable(r, m, itt) for i, r in enumerate(design)]
        pos = [i for i, r in enumerate(design) if ok[i] and m in r["memes"] and r["version"] in condition]
        if scheme == "hard":
            neg = [i for i, r in enumerate(design) if ok[i] and m in r["memes"] and r["version"] in ("control", "decoy")]
        elif scheme == "matched":
            neg = [i for i, r in enumerate(design) if ok[i] and m not in r["memes"] and r["version"] in condition]
        else:
            neg = [i for i, r in enumerate(design) if ok[i] and not _plants(r, m)]
        out[m] = auc(col[pos], col[neg])
    return out


def paired_meme_test(auc_a, auc_b, memes, alternative="greater", null_value=0.0, sesoi=None):
    d = np.array([auc_a.get(m, np.nan) - auc_b.get(m, np.nan) for m in memes], float)
    r = signflip(d, alternative=alternative, null_value=null_value)
    r["mean_auc_a"] = float(np.nanmean([auc_a.get(m, np.nan) for m in memes]))
    r["mean_auc_b"] = float(np.nanmean([auc_b.get(m, np.nan) for m in memes]))
    r.update(mean_ci(d, key=f"memepair{len(d)}"))
    if sesoi is not None:
        r["falsification"] = falsification_test(d, sesoi)
    r["per_meme_d"] = dict(zip(memes, d.tolist()))
    return r


def plot_flip_test(Ma, Mb, design, memes, condition, valid=None, alternative="greater", null_value=0.0, n_mc=20000, itt=False, key="plotflip"):
    """Plot-level randomisation for a T2 method comparison: the same statistic as paired_meme_test (mean over memes of
    the twin-AUC difference), but every plot's pairs, across its three memes, flip sign together. It is valid when
    plots, not memes, carry the dependence (a non-specific detector scores the three memes of a plot alike). The
    primary T2 tests require both the meme-level and the plot-level test (the larger p is used)."""
    Ma, Mb = np.asarray(Ma, float), np.asarray(Mb, float)
    plots = sorted({r["plot"] for r in design})
    pid = {p: k for k, p in enumerate(plots)}
    W = np.zeros((len(memes), len(plots)))
    ok = np.zeros(len(memes), bool)
    for j, m in enumerate(memes):
        prs = twin_pairs(design, m, condition, valid, itt)
        ia, pa = _pair_ind(Ma[:, j], prs)
        ib, _ = _pair_ind(Mb[:, j], prs)
        if len(ia) and len(ia) == len(ib):
            d = ia - ib - null_value
            for dd, pp in zip(d, pa):
                W[j, pid[pp]] += dd / len(d)
            ok[j] = True
    w = W[ok].sum(0) / max(ok.sum(), 1)
    obs = float(w.sum())
    rng = rng_for(key, len(plots))
    null = np.concatenate([[obs], (rng.choice([-1.0, 1.0], size=(n_mc, len(plots))) * w).sum(1)])
    if alternative == "greater":
        p = float(np.mean(null >= obs - 1e-12))
    elif alternative == "less":
        p = float(np.mean(null <= obs + 1e-12))
    else:
        p = float(np.mean(np.abs(null) >= abs(obs) - 1e-12))
    return {"stat": obs + null_value, "p": p, "n_plots": int((np.abs(W[ok]).sum(0) > 0).sum())}


def two_way_bootstrap(Ma, Mb, design, memes, condition, valid=None, n=2000, key="twoway"):
    """95% interval for mean over memes of twin AUC(A) - twin AUC(B), resampling memes and plots (two-way, crossed)."""
    pairs = {m: twin_pairs(design, m, condition, valid) for m in memes}
    plots = sorted({r["plot"] for r in design})
    pid = {p: k for k, p in enumerate(plots)}
    per = []
    for j, m in enumerate(memes):
        ia, pa = _pair_ind(np.asarray(Ma, float)[:, j], pairs[m])
        ib, _ = _pair_ind(np.asarray(Mb, float)[:, j], pairs[m])
        if len(ia) and len(ia) == len(ib):
            per.append((ia - ib, np.array([pid[p] for p in pa])))
    if not per:
        return {"ci95": [float("nan")] * 2}
    rng = rng_for(key, len(per))
    vals = []
    for _ in range(n):
        cm = np.bincount(rng.integers(0, len(per), len(per)), minlength=len(per))
        cp = np.bincount(rng.integers(0, len(plots), len(plots)), minlength=len(plots))
        num = den = 0.0
        for k, (dd, pp) in enumerate(per):
            if cm[k] == 0:
                continue
            wts = cp[pp]
            if wts.sum() == 0:
                continue
            num += cm[k] * float((dd * wts).sum() / wts.sum())
            den += cm[k]
        if den:
            vals.append(num / den)
    return {"ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
            "lower95_one_sided": float(np.percentile(vals, 5))}


def specificity_matrix(M_det, det_memes, design, memes, condition, valid=None, itt=False, pairs_cache=None):
    """A[j, k] = twin AUC of detector j (built for det_memes[j]) on meme k's pairs, leaving out the plots where
    detector j's own meme is also planted (so a detector is never credited for its own meme in another's plot)."""
    M_det = np.asarray(M_det, float)
    plants = defaultdict(set)
    for r in design:
        for m in r["memes"]:
            plants[m].add(r["plot"])
    A = np.full((len(det_memes), len(memes)), np.nan)
    for j, dm in enumerate(det_memes):
        for k, m in enumerate(memes):
            skip = plants[dm] if m != dm else set()
            key = (m, dm if m != dm else None)
            if pairs_cache is not None and key in pairs_cache:
                prs = pairs_cache[key]
            else:
                prs = twin_pairs(design, m, condition, valid, itt, skip_plots=skip)
                if pairs_cache is not None:
                    pairs_cache[key] = prs
            ind, _ = _pair_ind(M_det[:, j], prs)
            A[j, k] = ind.mean() if len(ind) else np.nan
    return A


def specificity_test(A, det_memes, memes, n_perm=20000, rng=None, sesoi=None):
    """H3: specificity of each DEV unit = its twin AUC on its own meme minus its mean twin AUC on every other meme. A
    unit that fires on 'something dramatic happens' scores the same on every meme, so its specificity is 0. Statistic:
    the mean over units. Null: the meme labels are permuted across the detectors (each detector is assigned another
    detector's meme), 20,000 times; one-sided p = P(null >= observed)."""
    rng = rng or rng_for("specificity", len(det_memes))
    idx = np.array([memes.index(m) for m in det_memes])
    A = np.asarray(A, float)

    def spec(assign):
        s = []
        for j, k in enumerate(assign):
            others = np.delete(A[j], k)
            s.append(A[j, k] - np.nanmean(others))
        return np.array(s)
    sp = spec(idx)
    obs = float(np.nanmean(sp))
    null = np.array([np.nanmean(spec(idx[rng.permutation(len(idx))])) for _ in range(n_perm)])
    r = {"stat": obs, "p": float((np.sum(null >= obs - 1e-12) + 1) / (n_perm + 1)), "n": int(np.isfinite(sp).sum()),
         "null_mean": float(null.mean()), "per_unit_specificity": dict(zip(det_memes, sp.tolist())),
         "per_unit_own_auc": dict(zip(det_memes, [float(A[j, k]) for j, k in enumerate(idx)]))}
    r.update(mean_ci(sp, key="spec"))
    if sesoi is not None:
        r["falsification"] = falsification_test(sp, sesoi)
    return r


def precision_at_r(scores, design, memes, condition, valid=None):
    """R-precision on the all-negatives set (secondary)."""
    scores = np.asarray(scores, float)
    valid = np.ones(len(design), bool) if valid is None else np.asarray(valid)
    out = {}
    for j, m in enumerate(memes):
        pos = [i for i, r in enumerate(design) if valid[i] and _usable(r, m) and m in r["memes"] and r["version"] in condition]
        neg = [i for i, r in enumerate(design) if valid[i] and _usable(r, m) and not _plants(r, m)]
        if not pos:
            out[m] = float("nan")
            continue
        ids = np.array(pos + neg)
        lab = np.array([1] * len(pos) + [0] * len(neg))
        order = np.argsort(-np.nan_to_num(scores[ids, j], nan=-np.inf), kind="stable")
        out[m] = float(lab[order[:len(pos)]].mean())
    return out


def f1_at_bar(scores, design, memes, condition, bar, valid=None):
    """F1 when a meme is called present if the story score beats the method's presence bar (second-highest score
    among the reference films; LLM judges and NLI: 0.5)."""
    scores = np.asarray(scores, float)
    valid = np.ones(len(design), bool) if valid is None else np.asarray(valid)
    out = {}
    for j, m in enumerate(memes):
        pos = np.array([bool(valid[i]) and _usable(r, m) and m in r["memes"] and r["version"] in condition for i, r in enumerate(design)])
        neg = np.array([bool(valid[i]) and _usable(r, m) and not _plants(r, m) for i, r in enumerate(design)])
        call = np.nan_to_num(scores[:, j], nan=-np.inf) > bar[j]
        tp, fp, fn = (call & pos).sum(), (call & neg).sum(), (~call & pos).sum()
        out[m] = float(2 * tp / max(2 * tp + fp + fn, 1))
    return out


# =============================================================================================== T3 helpers
def stratified_auc(score, positives, negatives_by_stratum, stratum_of, entity_of=None):
    """Mean over positives of the share of same-stratum negatives they outscore (ties one half); with entity_of each
    entity counts once."""
    vals, ents = [], []
    for p in positives:
        neg = negatives_by_stratum.get(stratum_of[p], [])
        if not len(neg):
            continue
        vals.append(auc([score[p]], np.asarray(score)[np.asarray(neg)]))
        ents.append(entity_of[p] if entity_of else p)
    vals, ents = np.array(vals), np.array(ents, dtype=object)
    if not len(vals):
        return float("nan"), [], []
    keys = list(dict.fromkeys(ents.tolist()))
    per = np.array([vals[ents == k].mean() for k in keys])
    return float(np.nanmean(per)), per.tolist(), keys


def entity_level_test(score, positives, entity_of, negatives_by_stratum, stratum_of, neg_cluster_of, n_perm=10000, rng=None):
    """T3 inference with franchise clustering (review round 1). Units are franchises: each positive entity, and each
    title-stem cluster of >= 2 pool negatives (pseudo-franchises). A unit's score = the mean over its works of the
    work's within-stratum percentile among the pool negatives, leaving out the unit's own works. Statistic: the mean
    score of the positive entities. Null: the positive labels are permuted among all units (positive entities and
    pseudo-franchises), so a franchise of near-duplicate works counts once on both sides. One-sided p = P(null >=
    observed). Also: a sign test of entity scores above 0.5 and each entity's score with a works-bootstrap interval."""
    rng = rng or rng_for("entity", len(positives))
    score = np.asarray(score, float)
    ents = defaultdict(list)
    for p in positives:
        if negatives_by_stratum.get(stratum_of[p]):
            ents[entity_of[p]].append(p)
    if not ents:
        return {"stat": float("nan"), "p": float("nan"), "n_entities": 0}

    def pct(i, exclude=()):
        neg = [j for j in negatives_by_stratum[stratum_of[i]] if j not in exclude]
        return auc([score[i]], score[np.asarray(neg)]) if neg else np.nan
    ent_score = {e: float(np.nanmean([pct(i) for i in ws])) for e, ws in ents.items()}
    clusters = defaultdict(list)
    for s_, negs in negatives_by_stratum.items():
        for i in negs:
            if neg_cluster_of.get(i):
                clusters[neg_cluster_of[i]].append(i)
    clusters = {c: v for c, v in clusters.items() if len(v) >= 2}
    cl_score = np.array([np.nanmean([pct(i, set(v)) for i in v]) for c, v in sorted(clusters.items())])
    k = len(ent_score)
    units = np.r_[np.array(list(ent_score.values())), cl_score]
    obs = float(np.mean(list(ent_score.values())))
    if len(cl_score):
        null = np.array([units[rng.permutation(len(units))[:k]].mean() for _ in range(n_perm)])
        p_ = float((np.sum(null >= obs - 1e-12) + 1) / (n_perm + 1))
    else:
        null, p_ = np.full(1, np.nan), float("nan")
    per = {}
    for e, ws in ents.items():
        v = np.array([pct(i) for i in ws])
        bs = [np.nanmean(v[rng.integers(0, len(v), len(v))]) for _ in range(1000)] if len(v) > 1 else [v[0]] * 2
        per[e] = {"score": ent_score[e], "n_works": len(ws), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    kk = int(sum(v > 0.5 for v in ent_score.values()))
    return {"stat": obs, "p": p_, "null_mean": float(np.nanmean(null)), "n_entities": k, "n_null_clusters": len(cl_score),
            "sign_test_p": float(sps.binomtest(kk, k, 0.5, alternative="greater").pvalue), "per_entity": per}


def entity_signflip(per_a, per_b, alternative="greater"):
    """Entity-level head-to-head of two methods' entity scores (T3 directional contrasts)."""
    common = [e for e in per_a if e in per_b]
    d = np.array([per_a[e]["score"] - per_b[e]["score"] for e in common])
    r = signflip(d, alternative=alternative)
    r.update(mean_ci(d, key="entityflip"))
    r["entities"] = common
    return r


def entity_weights(positives, entity_of):
    ents = [entity_of[p] for p in positives]
    c = Counter(ents)
    return np.array([1.0 / (len(c) * c[e]) for e in ents])


def stratified_perm_test(score, positives, weights, negatives_by_stratum, stratum_of, n_perm=10000, rng=None):
    """The draft's work-level T3 test (secondary only: it treats works of one franchise as independent)."""
    rng = rng or rng_for("t3perm", len(positives))
    score = np.asarray(score, float)
    by = defaultdict(list)
    for p, w in zip(positives, weights):
        if negatives_by_stratum.get(stratum_of[p]):
            by[stratum_of[p]].append((p, w))

    def stat_for(pos_scores, neg_scores, ws):
        srt = np.sort(neg_scores)
        less = np.searchsorted(srt, pos_scores, side="left")
        eq = np.searchsorted(srt, pos_scores, side="right") - less
        return float(((less + 0.5 * eq) / len(srt)) @ ws)
    obs, parts = 0.0, []
    for s, items in by.items():
        ps = np.array([p for p, _ in items])
        ws = np.array([w for _, w in items])
        neg = np.asarray(negatives_by_stratum[s])
        parts.append((score[np.concatenate([ps, neg])], len(ps), ws))
        obs += stat_for(score[ps], score[neg], ws)
    tot_w = sum(p[2].sum() for p in parts)
    null = np.zeros(n_perm)
    for i in range(n_perm):
        v = 0.0
        for allv, k, ws in parts:
            perm = rng.permutation(len(allv))
            v += stat_for(allv[perm[:k]], allv[perm[k:]], ws)
        null[i] = v
    return {"stat": obs / tot_w, "null_mean": float(null.mean() / tot_w),
            "p": float((np.sum(null >= obs - 1e-12) + 1) / (n_perm + 1)), "n_positives": int(sum(p[1] for p in parts))}


def centroid_likeness(V, ref_rows, centre_rows, loo_rows=()):
    """cos(V - mu, mean(V[ref]) - mu), mu = mean of the centring rows; rows in loo_rows (members of ref_rows) are
    compared with the centroid of the other ref rows (leave-one-out)."""
    V = np.asarray(V, np.float64)
    mu = V[centre_rows].mean(0)
    X = V - mu
    ref_rows = list(ref_rows)
    s = X[ref_rows].sum(0)
    n = len(ref_rows)
    c = s / n
    out = (X @ c) / (np.linalg.norm(X, axis=1) * np.linalg.norm(c) + 1e-12)
    for i in loo_rows:
        ci = (s - X[i]) / (n - 1)
        out[i] = float(X[i] @ ci / (np.linalg.norm(X[i]) * np.linalg.norm(ci) + 1e-12))
    return out


def residualise(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    ok = np.isfinite(y) & np.isfinite(x)
    X = np.column_stack([np.ones(ok.sum()), x[ok]])
    beta = np.linalg.lstsq(X, y[ok], rcond=None)[0]
    out = np.full(len(y), np.nan)
    out[ok] = y[ok] - X @ beta
    return out


# =============================================================================================== T4 helpers
def mutation_stats(P_ordered, skip=None):
    """For presence [n_works (chronological), n_units]: number of mutations per work (units first present there),
    each mutation's success = the number of later works that carry it, leaving out the later works in skip[i] (the
    documented adaptations of the innovating work i), and each work's mean success = mean over its mutations of
    success / number of counted later works."""
    P_ordered = np.asarray(P_ordered, bool)
    n = P_ordered.shape[0]
    any_ = P_ordered.any(0)
    fa = np.argmax(P_ordered, axis=0)
    fa[~any_] = -1
    later = np.zeros(P_ordered.shape[1])
    n_mut = np.zeros(n, int)
    mean_succ = np.full(n, np.nan)
    for i in range(n):
        u = np.where(fa == i)[0]
        n_mut[i] = len(u)
        if not len(u) or i == n - 1:
            continue
        rows = [k for k in range(i + 1, n) if not skip or k not in skip.get(i, ())]
        if not rows:
            continue
        cnt = P_ordered[rows][:, u].sum(0)
        later[u] = cnt
        mean_succ[i] = float((cnt / len(rows)).mean())
    return n_mut, mean_succ, later, fa


def _skip_for(order, adapt_of):
    """Map chronological positions: skip[i] = positions of the documented adaptations of the work at position i."""
    if not adapt_of:
        return None
    pos = {int(w): k for k, w in enumerate(order)}
    return {pos[w]: {pos[a] for a in adapt_of.get(int(w), ()) if a in pos} for w in order}


def innovators_test(P, order, trio_ids, n_perm=2000, rng=None, adapt_of=None):
    """S18. Statistic: mean success of the trio's mutations, counted only in later works that are not documented
    adaptations of the innovating work. Null: position-preserving shuffles of which work occupies each slot (the
    exclusion travels with the work). p = P(null >= observed)."""
    rng = rng or rng_for("t4", len(order))
    order = np.asarray(order)
    slots = [int(np.where(order == t)[0][0]) for t in trio_ids]

    def stat(ordr):
        _, ms, _, _ = mutation_stats(np.asarray(P[ordr], bool), _skip_for(ordr, adapt_of))
        return float(np.nan_to_num(ms[slots], nan=0.0).mean())
    obs = stat(order)
    null = np.array([stat(rng.permutation(order)) for _ in range(n_perm)])
    return {"stat": obs, "null_mean": float(np.nanmean(null)), "p": float((np.sum(null >= obs - 1e-12) + 1) / (n_perm + 1)), "slots": slots}


def curveball(P, n_trades, rng):
    """Margin-preserving swap randomisation of a binary matrix (Strona et al. 2014): every row and column total is
    kept; returns a new boolean matrix."""
    P = np.asarray(P, bool)
    rows = [np.flatnonzero(r) for r in P]
    n = len(rows)
    for _ in range(n_trades):
        a, b = rng.choice(n, 2, replace=False)
        A_, B_ = rows[a], rows[b]
        a_only = np.setdiff1d(A_, B_, assume_unique=True)
        b_only = np.setdiff1d(B_, A_, assume_unique=True)
        if not len(a_only) or not len(b_only):
            continue
        both = np.intersect1d(A_, B_, assume_unique=True)
        pool = rng.permutation(np.concatenate([a_only, b_only]))
        rows[a] = np.sort(np.concatenate([both, pool[:len(a_only)]]))
        rows[b] = np.sort(np.concatenate([both, pool[len(a_only):]]))
    out = np.zeros_like(P)
    for i, r in enumerate(rows):
        out[i, r] = True
    return out


def selection_null(P_ordered, skip, n_sims=500, trades_per_row=5, rng=None):
    """S19 null that keeps every unit's frequency and every work's unit count: curveball randomisation of the
    work-by-unit matrix with the time order shuffled. Returns the null Gini of mutation success."""
    rng = rng or rng_for("selnull", P_ordered.shape)
    P_ordered = np.asarray(P_ordered, bool)
    keep = P_ordered.any(0)
    Q = P_ordered[:, keep]
    out = []
    for _ in range(n_sims):
        R = curveball(Q, trades_per_row * Q.shape[0], rng)[rng.permutation(Q.shape[0])]
        _, _, later, fa = mutation_stats(R, None)
        mut = later[(fa >= 0) & (fa < len(R) - 1)]
        out.append(gini(mut))
    return np.array(out)


def selection_excess(P_ordered, skip, n_sims=500, rng=None):
    """Observed Gini of mutation success (adaptations of the innovating work excluded) against the margin-preserving
    null; the standardised excess z = (obs - null mean) / null SD."""
    _, _, later, fa = mutation_stats(P_ordered, skip)
    mut = later[(fa >= 0) & (fa < len(P_ordered) - 1)]
    g = gini(mut)
    null = selection_null(P_ordered, skip, n_sims, rng=rng)
    sd = float(null.std()) + 1e-12
    return {"gini_obs": g, "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
            "z": float((g - null.mean()) / sd), "p": float((np.sum(null >= g) + 1) / (len(null) + 1)), "null": null}


def compare_excess(ex_a, ex_b):
    """S19 comparative: is representation A's standardised excess over its null larger than B's? p from the two
    independent null distributions (the difference of standardised null draws)."""
    za = (ex_a["null"] - ex_a["null"].mean()) / (ex_a["null"].std() + 1e-12)
    zb = (ex_b["null"] - ex_b["null"].mean()) / (ex_b["null"].std() + 1e-12)
    m = min(len(za), len(zb))
    obs = ex_a["z"] - ex_b["z"]
    null = za[:m] - zb[:m]
    return {"stat": float(obs), "p": float((np.sum(null >= obs) + 1) / (m + 1))}


def neutral_copying(n_units_per_work, n_mut_per_work, n_sims=500, rng=None):
    """The draft's Bentley-style random-copying model (secondary). Returns [n_sims, 2]: Gini of mutation success and
    the share of mutations carried by >= 2 later works."""
    rng = rng or rng_for("neutral", len(n_units_per_work))
    n = len(n_units_per_work)
    total = int(np.sum(n_mut_per_work[1:])) + int(n_units_per_work[0])
    out = []
    for _ in range(n_sims):
        counts = np.zeros(total)
        Pm = np.zeros((n, total), bool)
        nxt = 0
        for t in range(n):
            m = int(n_units_per_work[0]) if t == 0 else int(n_mut_per_work[t])
            k = int(n_units_per_work[t]) - m
            if t > 0 and k > 0 and counts.sum() > 0:
                p = np.minimum(1.0, k * counts / counts.sum())
                Pm[t] = rng.random(total) < p
            Pm[t, nxt:nxt + m] = True
            nxt += m
            counts += Pm[t]
        _, _, later, fa = mutation_stats(Pm)
        mut = later[(fa >= 0) & (fa < n - 1)]
        out.append((gini(mut), float(np.mean(mut >= 2)) if len(mut) else 0.0))
    return np.array(out)


def batch_effect_auc(P, labels, years, lo=1995, hi=2022):
    """T4 batch-effect check: leave-one-out AUC of an L2 logistic regression separating DEV-origin from TEST-origin
    works dated lo-hi on unit presence. Above 0.7 flags T4."""
    from sklearn.linear_model import LogisticRegression
    idx = [i for i, y in enumerate(years) if isinstance(y, (int, np.integer)) and lo <= y <= hi]
    X = np.asarray(P, float)[idx]
    y = np.array([labels[i] == "dev" for i in idx], int)
    if len(set(y)) < 2:
        return {"auc": float("nan"), "n": len(idx)}
    keep = X.std(0) > 0
    X = X[:, keep]
    sc = np.zeros(len(idx))
    for k in range(len(idx)):
        tr = np.delete(np.arange(len(idx)), k)
        if len(set(y[tr])) < 2:
            sc[k] = 0.5
            continue
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], y[tr])
        sc[k] = lr.decision_function(X[k:k + 1])[0]
    return {"auc": auc(sc[y == 1], sc[y == 0]), "n": len(idx), "n_dev": int(y.sum())}


# =============================================================================================== T5 helpers
def corr_cols(X, Y):
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)
    return (X * Y).mean(0)


def best_partner(q_profiles, partner_matrix, chunk=20000):
    """For each column of q_profiles [n_sent, k], the column of partner_matrix [n_sent, U] with the highest Pearson r."""
    Q = np.asarray(q_profiles, np.float64)
    Q = (Q - Q.mean(0)) / (Q.std(0) + 1e-12)
    n = Q.shape[0]
    best_r = np.full(Q.shape[1], -np.inf)
    best_u = np.zeros(Q.shape[1], int)
    for c0 in range(0, partner_matrix.shape[1], chunk):
        Y = np.asarray(partner_matrix[:, c0:c0 + chunk], np.float64)
        Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)
        R = Q.T @ Y / n
        j = R.argmax(1)
        r = R[np.arange(R.shape[0]), j]
        better = r > best_r
        best_r[better] = r[better]
        best_u[better] = j[better] + c0
    return best_u, best_r


def t5_test(r_concept, r_null, groups, n_perm=20000, rng=None):
    """T5: held-out twin correlation of concept memes vs their lexically matched null units; within each matched set
    the concept label is swapped with a random member."""
    rng = rng or rng_for("t5", len(r_concept))
    r_concept, r_null = np.asarray(r_concept), np.asarray(r_null)
    sets = [np.concatenate([[r_concept[i]], r_null[g]]) for i, g in enumerate(groups)]

    def stat(pick):
        return float(np.mean([s[p] - np.delete(s, p).mean() for s, p in zip(sets, pick)]))
    obs = stat([0] * len(sets))
    null = np.array([stat([rng.integers(0, len(s)) for s in sets]) for _ in range(n_perm)])
    return {"stat": obs, "p": float((np.sum(null >= obs - 1e-12) + 1) / (n_perm + 1)), "n": len(sets)}


# =============================================================================================== feature loading
class Set:
    """One text set with its features (lazy)."""

    def __init__(self, name, data_dir=None, feat_dir=None):
        self.name = name
        self.data_dir = Path(data_dir or DATA)
        self.feat = Path(feat_dir or FEAT) / name
        self.works = read_jsonl(self.data_dir / f"{name}.jsonl")
        self.ids = [w["id"] for w in self.works]
        self.index = {w: i for i, w in enumerate(self.ids)}
        self.sentences = [s for w in self.works for s in w["sentences"]]
        self.sent_work = np.array([i for i, w in enumerate(self.works) for _ in w["sentences"]], int)

    def texts(self):
        return [lexical_text(w["sentences"]) for w in self.works]

    def has(self, rel):
        return (self.feat / rel).exists()

    def agg(self, model):
        A_ = np.load(self.feat / f"neurons_{model}" / "agg.npy", mmap_mode="r")
        meta = json.load(open(self.feat / f"neurons_{model}" / "meta.json"))
        assert meta["work_ids"] == self.ids, f"{self.name}: neuron work order differs from the texts file"
        return A_

    def sent_units(self, model):
        d = self.feat / f"neurons_{model}"
        return np.load(d / "sent.npy", mmap_mode="r"), np.load(d / "units.npy")

    def full(self, model):
        return np.load(self.feat / f"neurons_{model}" / "full.npy", mmap_mode="r")

    def hidden(self, model):
        return np.load(self.feat / f"neurons_{model}" / "hidden.npy", mmap_mode="r")

    def emb(self, enc, unit="sentence"):
        suffix = {"sentence": "", "win3": "_win3", "doc": "_doc"}[unit]
        E = np.load(self.feat / f"emb_{enc}{suffix}.npy", mmap_mode="r")
        want = len(self.ids) if unit == "doc" else len(self.sentences)
        assert E.shape[0] == want, f"{self.name}: {unit} embedding rows differ"
        return E

    def llm(self, reader, task):
        p = self.feat / f"llm_{reader}_{task}.jsonl"
        return read_jsonl(p) if p.exists() else None

    def nli(self, model="deberta-v3-base"):
        p = self.feat / f"nli_{model}.npy"
        return np.load(p) if p.exists() else None

    def cost(self, method):
        p = self.feat / f"cost_{method.replace(':', '_')}.json"
        return json.load(open(p)) if p.exists() else None

    def word_counts(self):
        return np.array([n_words(w["sentences"]) for w in self.works], float)


def _maybe(name):
    return Set(name) if (DATA / f"{name}.jsonl").exists() else None


def work_embeddings(S, enc):
    E = np.asarray(S.emb(enc), np.float32)
    W = work_from_sentences(E, S.sent_work, len(S.ids), how="mean")
    return W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)


def _logit(row):
    if "logit" in row and row["logit"] is not None:
        return float(row["logit"])
    p = min(max(float(row["p_yes"]), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def code_features(sentences):
    """Codes-only baseline features: number of distinct codes, codes per sentence and the 10-bin profile of where
    codes fall in the text (relative sentence position)."""
    k = max(len(sentences), 1)
    codes = [CODE.findall(s) for s in sentences]
    distinct = len({c for cs in codes for c in cs})
    per = sum(len(cs) for cs in codes) / k
    hist = np.zeros(10)
    for i, cs in enumerate(codes):
        hist[min(9, int(10 * i / k))] += len(cs)
    hist = hist / max(hist.sum(), 1)
    return np.r_[math.log1p(distinct), per, hist]


# =============================================================================================== lineage similarity
LINEAGE_GRID = {"neurons": ["base", "idf", "q", "cos"], "words": ["base", "idf", "q"], "tfidf": ["base", "ng12", "df1", "nosub"],
                "lsa": ["k100", "k50", "k200"], "lda": ["k30", "k20", "k50"], "emb": ["mean", "centred", "align", "presence"],
                "hidden": ["L1", "L0", "L2", "L3"], "topics": ["mcs15", "mcs10", "mcs30"], "llmsum": ["base"],
                "llmfull": ["base"], "codes": ["base"], "length": ["base"]}


def _family(m):
    name, _, cfg = m.partition("@")
    kind, _, arg = name.partition(":")
    return name, kind, arg, cfg


def resolve(m, selected=None):
    """'tfidf' -> 'tfidf@<DEV-selected config>' (the default config when nothing was selected); 'x@cfg' stays."""
    name, kind, arg, cfg = _family(m)
    if cfg:
        return m
    sel = (selected or {}).get(name)
    return f"{name}@{sel or LINEAGE_GRID[kind][0]}"


def lineage_similarities(row_groups, cand_groups, ref_group, lineage_groups=None, methods=None, cand_ref_group=None):
    """Similarity matrices [n_rows, n_cands] for lineage methods given as 'family@config' (resolve first).
    row_groups / cand_groups: lists of (Set, indices). ref_group: (Set, indices) of the reference films that set the
    presence bars and are added to the lexical fitting texts. cand_ref_group: reference films in the candidates'
    language (T1c: the comics' English bars) when it differs. lineage_groups: the works that define optional bands."""
    methods = methods or []
    lineage_groups = lineage_groups or (list(row_groups) + list(cand_groups))
    out = {}

    def texts(groups):
        return [_texts(S)[i] for S, ix in groups for i in ix]
    rt, ct, rf, lt = texts(row_groups), texts(cand_groups), texts([ref_group]), texts(lineage_groups)
    if cand_ref_group is not None:
        rf = rf + texts([cand_ref_group])
    nr, nc = len(rt), len(ct)
    fit_texts = rt + ct + rf
    tag = row_groups[0][0].name
    cache = {}

    def tf(cfg):
        if cfg not in cache:
            kw = {"base": {}, "ng12": {"ngram": (1, 2)}, "df1": {"min_df": 1}, "nosub": {"sublinear": False}}[cfg]
            cache[cfg] = tfidf_matrix(rt + ct, fit_texts, **kw)
        return cache[cfg]
    for m in methods:
        name, kind, arg, cfg = _family(m)
        t0 = time.time()
        try:
            if kind == "tfidf":
                X, _ = tf(cfg)
                out[m] = cosine(X[:nr].toarray(), X[nr:].toarray())
            elif kind == "lsa":
                X, vec = tf("base")
                L = lsa_vectors(vec.transform(fit_texts), X, k=int(cfg[1:]))
                out[m] = cosine(L[:nr], L[nr:])
            elif kind == "words":
                X, vec = tf("base")
                Pw = (X > 0).toarray()
                Pl = (vec.transform(lt) > 0).toarray()
                opt = optional_band(Pl, frac=0.25 if cfg == "q" else 0.5)
                out[m] = containment(Pw, range(nr), range(nr, nr + nc), opt, unit_idf(Pl) if cfg == "idf" else None)
            elif kind == "lda":
                th, _ = lda_fit_transform(fit_texts, rt + ct, K=int(cfg[1:]), seed=0)
                out[m] = cosine(th[:nr], th[nr:])
            elif kind == "length":
                ln = np.log1p([len(t.split()) for t in rt + ct])
                out[m] = -np.abs(ln[:nr, None] - ln[None, nr:])
            elif kind == "codes":
                F = np.array([code_features(S.works[i]["sentences"]) for S, ix in list(row_groups) + list(cand_groups) for i in ix])
                Fr = np.array([code_features(S.works[i]["sentences"]) for S, ix in [ref_group] for i in ix])
                mu, sd = Fr.mean(0), Fr.std(0) + 1e-6
                Z = (F - mu) / sd
                out[m] = cosine(Z[:nr], Z[nr:])
            else:
                _model_side_similarity(m, kind, arg, cfg, out, row_groups, cand_groups, ref_group, lineage_groups, nr, nc, cand_ref_group)
        except (FileNotFoundError, KeyError, AssertionError) as err:
            out[m] = {"missing": str(err)}
        TIMING[tag][f"similarity {m}"] = time.time() - t0
    return out


_TEXT_CACHE = {}


def _texts(S):
    key = (S.name, str(S.data_dir))
    if key not in _TEXT_CACHE or len(_TEXT_CACHE[key]) != len(S.ids):
        _TEXT_CACHE[key] = S.texts()
    return _TEXT_CACHE[key]


def _agg(groups, model):
    return np.concatenate([np.asarray(S.agg(model), np.float32)[list(ix)] for S, ix in groups])


def _model_side_similarity(m, kind, arg, cfg, out, row_groups, cand_groups, ref_group, lineage_groups, nr, nc, cand_ref_group):
    if kind == "neurons":
        Ar, Ac, Aref = _agg(row_groups, arg), _agg(cand_groups, arg), _agg([ref_group], arg)
        if cfg == "cos":
            med, sd = unit_scale(Aref)
            out[m] = cosine((Ar - med) / sd, (Ac - med) / sd)
            return
        bar_r = presence_bar(Aref)
        bar_c = presence_bar(_agg([cand_ref_group], arg)) if cand_ref_group is not None else bar_r
        cand_sets = {S.name for S, _ in cand_groups}
        Pl = np.concatenate([np.asarray(S.agg(arg), np.float32)[list(ix)] > (bar_c if S.name in cand_sets else bar_r)
                             for S, ix in lineage_groups])
        opt = optional_band(Pl, frac=0.25 if cfg == "q" else 0.5)
        Pall = np.concatenate([Ar > bar_r, Ac > bar_c])
        out[m] = containment(Pall, range(nr), range(nr, nr + nc), opt, unit_idf(Pl) if cfg == "idf" else None)
    elif kind == "emb":
        if cfg in ("mean", "centred"):
            def we(groups):
                return np.concatenate([work_embeddings(S, arg)[list(ix)] for S, ix in groups])
            Wr, Wc = we(row_groups), we(cand_groups)
            if cfg == "centred":
                mu = we([ref_group]).mean(0)
                Wr, Wc = Wr - mu, Wc - mu
            out[m] = cosine(Wr, Wc)
        elif cfg == "align":
            def sents(groups):
                res = []
                for S, ix in groups:
                    E = np.asarray(S.emb(arg), np.float32)
                    res += [E[S.sent_work == i] for i in ix]
                return res
            R, Cs = sents(row_groups), sents(cand_groups)
            M = np.zeros((nr, nc))
            for j, c in enumerate(Cs):
                for i, r in enumerate(R):
                    M[i, j] = float((c @ r.T).max(1).mean()) if len(c) and len(r) else 0.0
            out[m] = M
        else:      # 'presence': the neuron rule applied to embedding dimensions (top-3 pooled, both poles)
            def pooled(groups):
                res = []
                for S, ix in groups:
                    E = np.asarray(S.emb(arg), np.float32)
                    E2 = np.concatenate([E, -E], 1)
                    res.append(work_from_sentences(E2, S.sent_work, len(S.ids))[list(ix)])
                return np.concatenate(res)
            Pr, Pc, Pref = pooled(row_groups), pooled(cand_groups), pooled([ref_group])
            bar = presence_bar(Pref)
            Pl = pooled(lineage_groups) > bar
            out[m] = containment(np.concatenate([Pr > bar, Pc > bar]), range(nr), range(nr, nr + nc), optional_band(Pl))
    elif kind == "topics":
        parts, owner = [], []
        for gi, (S, ix) in enumerate(list(row_groups) + list(cand_groups)):
            E = np.asarray(S.emb(arg), np.float32)
            for k, i in enumerate(ix):
                sel = np.where(S.sent_work == i)[0]
                parts.append(E[sel])
                owner += [(gi, k)] * len(sel)
        lab, _, _, _, _ = cluster_topics(np.concatenate(parts), min_cluster_size=int(cfg[3:]))
        K = max(int(lab.max()) + 1, 1)
        offsets, o = [], 0
        for S, ix in list(row_groups) + list(cand_groups):
            offsets.append(o)
            o += len(ix)
        Hh = np.zeros((nr + nc, K))
        for (gi, k), l in zip(owner, lab):
            if l >= 0:
                Hh[offsets[gi] + k, l] += 1
        out[m] = cosine(Hh[:nr] + 1e-9, Hh[nr:] + 1e-9)
    elif kind == "hidden":
        layer = int(cfg[1:])

        def wh(groups):
            rows = []
            for S, ix in groups:
                Hs = np.asarray(S.hidden(arg)[:, layer], np.float32)
                rows.append(work_from_sentences(Hs, S.sent_work, len(S.ids), how="mean")[list(ix)])
            return np.concatenate(rows)
        Hr, Hc, Href = wh(row_groups), wh(cand_groups), wh([ref_group])
        mu, sd = Href.mean(0), Href.std(0) + 1e-6
        out[m] = cosine((Hr - mu) / sd, (Hc - mu) / sd)
    elif kind in ("llmsum", "llmfull"):
        task = "pairs" if kind == "llmsum" else "pairsfull"
        P = {}
        for S, _ in row_groups:
            for p in S.llm(arg, task) or []:
                P[(p["a"], p["b"])] = _logit(p)
        rid = [S.ids[i] for S, ix in row_groups for i in ix]
        cid = [S.ids[i] for S, ix in cand_groups for i in ix]
        M = np.array([[P.get((a, b), np.nan) for b in cid] for a in rid])
        if np.isnan(M).all():
            raise FileNotFoundError(f"no {task} ratings for {m}")
        out[m] = M
    else:
        raise KeyError(m)


def lineage_methods(models=MODELS, encoders=(PRIMARY_ENCODER, "e5-large", "e5-base", "labse", "minilm"), llm=True, hidden=True,
                    full_reader=False, grid=False):
    """Lineage method families (grid=True: every configuration of every family, for the F1 selection)."""
    fam = [f"neurons:{x}" for x in models] + ["tfidf", "words", "lsa", "lda", "codes", "length"]
    fam += [f"emb:{e}" for e in encoders] + [f"topics:{PRIMARY_ENCODER}"]
    if hidden:
        fam += [f"hidden:{x}" for x in models]
    if llm:
        fam += [f"llmsum:{x}" for x in models]
    if full_reader:
        fam += [f"llmfull:{x}" for x in models]
    if not grid:
        return fam
    return [f"{f}@{c}" for f in fam for c in LINEAGE_GRID[_family(f)[1]]]


# =============================================================================================== DEV Batman metadata
# copied from experiments.py (labels as in data/corpus.jsonl)
BAT_ADAPTATIONS = {
    "Batman Begins": ["Year One"], "The Dark Knight": ["The Long Halloween", "The Killing Joke"],
    "Dark Knight Rises": ["Knightfall", "Dark Knight Returns"], "Batman v Superman": ["Dark Knight Returns"],
    "Under the Red Hood": ["A Death in the Family"], "Arkham Asylum (game)": ["Arkham Asylum (comic)"],
    "Batman '89": ["The Killing Joke"], "Joker (2019)": ["The Killing Joke"], "The Batman": ["Year One", "The Long Halloween"],
}
BAT_SCREEN_NONADAPT = ["1943 serial", "Batman '66", "Batman Returns", "Batman Forever", "Batman & Robin", "Mask of the Phantasm",
                       "Return of the Joker", "Arkham City", "Arkham Knight", "Lego Batman", "Batman Ninja"]
FRESH_ADAPTATIONS = {
    "Batman: Year One (film)": (2011, ["Year One"]), "Batman: The Dark Knight Returns (film)": (2012, ["Dark Knight Returns"]),
    "Batman: The Killing Joke (film)": (2016, ["The Killing Joke"]), "Batman: Hush (film)": (2019, ["Hush"]),
    "Batman: Death in the Family": (2020, ["A Death in the Family"]), "Batman: The Long Halloween (film)": (2021, ["The Long Halloween"]),
    "Batman: Knightfall (film)": (2026, ["Knightfall"]),
}
T4_TRIO = ("Dark Knight Returns", "Year One", "The Killing Joke")


def run_lineage(spec, methods, n_perm=N_PERM_LINEAGE, residual=True, keep_null=(), ties="half"):
    """lineage_test for every method on one prepared spec, plus the 'beyond words' residual test for the neuron
    methods (neuron similarity with TF-IDF cosine and word containment regressed out)."""
    res = {}
    for m in methods:
        if m not in spec["Sim"]:
            continue
        S = spec["Sim"][m]
        if isinstance(S, dict):
            res[m] = {"error": "features missing", "detail": S.get("missing")}
            continue
        if np.isnan(S).any():
            res[m] = {"error": "missing similarities", "n_missing": int(np.isnan(S).sum())}
            continue
        res[m] = lineage_test(S, spec["targets"], spec["sources"], spec["eligible"], spec["hub"], spec.get("weights"),
                              n_perm=n_perm, rng=rng_for("lineage", spec.get("name"), m), keep_null=m in keep_null, ties=ties)
    if residual:
        lex = [spec["Sim"][k] for k in ("tfidf@base", "words@base") if k in spec["Sim"] and not isinstance(spec["Sim"][k], dict)]
        for m in [k for k in methods if _family(k)[1] == "neurons"]:
            if m in spec["Sim"] and lex and not isinstance(spec["Sim"][m], dict) and not np.isnan(spec["Sim"][m]).any():
                R, r2 = residual_sim(spec["Sim"][m], lex, spec["fit_mask"])
                r = lineage_test(R, spec["targets"], spec["sources"], spec["eligible"], spec["hub"], spec.get("weights"),
                                 n_perm=n_perm, rng=rng_for("resid", spec.get("name"), m))
                r["r2_of_fit"] = r2
                res[f"{m}|resid_on_words"] = r
    return res


def _screen(S):
    return [i for i, w in enumerate(S.works) if w.get("group", "batman") == "batman" and w.get("medium") != "comic"]


def _comics(S):
    return [i for i, w in enumerate(S.works) if w.get("group") == "batman" and w.get("medium") == "comic"]


def _refs(S, drop=()):
    return [i for i, w in enumerate(S.works) if w.get("group") == "baseline" and w["id"] not in drop]


def assemble_batman_spec(name, screen_set, comics_set, fresh_set, ref_set, methods, fresh_only=False, cand_ref_set=None,
                         drop_targets=()):
    """The Batman lineage for one condition (E2 design; T1b level, T1c language, T1e fresh works). Rows = the 20 DEV
    screen works of screen_set then the fresh adaptations in fresh_set; candidates = the 12 DEV comics of comics_set
    (always English name-free); hub = the 11 screen works that adapt no comic; targets = the 9 DEV adaptations + the 7
    fresh ones (fresh_only: the 7); time rule; optional band from the condition's 32 Batman works; equal weights;
    clusters = first-listed documented source comic. cand_ref_set: reference films in the comics' language."""
    screen = _screen(screen_set)
    comics = _comics(comics_set)
    cid = [comics_set.ids[i] for i in comics]
    cyear = np.array([comics_set.works[i]["year"] for i in comics])
    fresh = [fresh_set.index[t] for t in FRESH_ADAPTATIONS if fresh_set is not None and t in fresh_set.index]
    row_groups = [(screen_set, screen)] + ([(fresh_set, fresh)] if fresh else [])
    lineage = [(screen_set, screen), (comics_set, comics)]
    cref = (cand_ref_set, _refs(cand_ref_set)) if cand_ref_set is not None else None
    Sim = lineage_similarities(row_groups, [(comics_set, comics)], (ref_set, _refs(ref_set)), lineage, methods, cref)
    row_of = {screen_set.ids[i]: k for k, i in enumerate(screen)}
    targets, sources, years, labels, clusters = [], [], [], [], []
    if not fresh_only:
        for t, srcs in BAT_ADAPTATIONS.items():
            if t in drop_targets or t not in row_of:
                continue
            targets.append(row_of[t])
            sources.append([cid.index(s) for s in srcs])
            years.append(screen_set.works[screen_set.index[t]]["year"])
            labels.append(t)
            clusters.append(f"batman:{srcs[0]}")
    for k, i in enumerate(fresh):
        t = fresh_set.ids[i]
        if t in drop_targets:
            continue
        targets.append(len(screen) + k)
        sources.append([cid.index(s) for s in FRESH_ADAPTATIONS[t][1]])
        years.append(FRESH_ADAPTATIONS[t][0])
        labels.append(t)
        clusters.append(f"batman:{FRESH_ADAPTATIONS[t][1][0]}")
    eligible = np.array([[cy <= y for cy in cyear] for y in years])
    hub = hub_groups_fixed(len(screen) + len(fresh), [row_of[h_] for h_ in BAT_SCREEN_NONADAPT if h_ in row_of], len(comics))
    return {"name": name, "Sim": Sim, "targets": targets, "sources": sources, "eligible": eligible, "hub": hub,
            "weights": None, "clusters": clusters, "fit_mask": np.ones((len(screen) + len(fresh), len(comics)), bool),
            "target_labels": labels}


def assemble_shakespeare_spec(name, films_set, plays_set, ref_set, methods, film_filter, row_filter=None,
                              drop_refs=("The Lion King",)):
    """T1d. Rows = the primary films passing row_filter (hub reference for play c = rows adapting other plays);
    targets = rows passing film_filter; candidates = all plays; play-weighted; clusters = source play."""
    films = [i for i, w in enumerate(films_set.works) if w.get("primary") and (row_filter is None or row_filter(w))]
    plays = list(range(len(plays_set.ids)))
    src = [[plays_set.ids.index(p) for p in films_set.works[i]["sources"]] for i in films]
    Sim = lineage_similarities([(films_set, films)], [(plays_set, plays)], (ref_set, _refs(ref_set, drop_refs)),
                               [(films_set, films), (plays_set, plays)], methods)
    tgt = [k for k, i in enumerate(films) if film_filter(films_set.works[i])]
    play_of = [films_set.works[films[k]]["sources"][0] for k in tgt]
    cnt = Counter(play_of)
    return {"name": name, "Sim": Sim, "targets": tgt, "sources": [src[k] for k in tgt],
            "eligible": np.ones((len(tgt), len(plays)), bool),
            "hub": hub_groups_other_targets(len(films), list(range(len(films))), src, len(plays)),
            "weights": [1.0 / (len(cnt) * cnt[p]) for p in play_of], "clusters": [f"play:{p}" for p in play_of],
            "fit_mask": np.ones((len(films), len(plays)), bool), "target_labels": [films_set.ids[films[k]] for k in tgt]}


def _shares_of(block, m):
    v = block.get(m, {})
    return v.get("per_target_share") if isinstance(v, dict) else None


def _recognised_ids(S, reader=PRIMARY_MODEL, titles=None):
    rows = S.llm(reader, "recognise") if S is not None else None
    if not rows:
        return None
    titles = titles or {}
    return {r["work"] for r in rows if recognised(r.get("answer"), titles.get(r["work"], r["work"]), None)}


def _drift_sets(lv):
    if lv == 0:
        return "dev_namefree", "t1e_fresh_namefree", "t1d_films_namefree"
    suf = "_style" if lv == "style" else str(lv)
    return f"t1b_drift{suf}", f"t1b_fresh_drift{suf}", f"t1b_shk_drift{suf}"


def run_t1(freeze, methods=None, n_perm=N_PERM_LINEAGE):
    """T1: lineage recovery under wording change. T1a (DEV, in dev_freeze), T1b vocabulary drift (H4), T1c Chinese,
    T1d Shakespeare (H5), T1e fresh Batman adaptations."""
    sel = freeze.get("lineage_selected", {})
    fams = methods or lineage_methods(full_reader=True)
    methods = sorted({resolve(m, sel) for m in fams} | {resolve(m) for m in fams if _family(m)[1] in ("tfidf", "lsa", "words")})
    nmeth = {x: resolve(f"neurons:{x}", sel) for x in MODELS}
    rivals = {r: resolve(r, sel) for r in LINEAGE_RIVALS}
    out = {"version": VERSION, "methods": methods, "selected": sel, "neuron_method": nmeth, "rival_methods": rivals}
    dev = Set("dev_namefree")
    keep = set(nmeth.values())
    # ---- T1b vocabulary drift: Batman arm and Shakespeare arm at every level (and the style-only control)
    drift, shk = {}, {}
    plays_nf = Set("t1d_plays_namefree")
    levels = list(DRIFT_LEVELS) + (["style"] if _maybe("t1b_drift_style") else [])
    in_drift = lambda w: w.get("drift_sample")
    for lv in levels:
        a, b, c = _drift_sets(lv)
        spec = assemble_batman_spec(f"t1b_{lv}", dev if lv == 0 else Set(a), dev, Set(b), dev, methods)
        drift[lv] = run_lineage(spec, methods, n_perm=n_perm, residual=False, keep_null=keep if lv in H4_CONJUNCT_LEVELS else ())
        drift[lv]["targets"], drift[lv]["clusters"] = spec["target_labels"], spec["clusters"]
        fs = Set(c)
        spec = assemble_shakespeare_spec(f"t1b_shk_{lv}", fs, plays_nf, dev, methods, in_drift, row_filter=in_drift)
        shk[lv] = run_lineage(spec, methods, n_perm=n_perm, residual=False, keep_null=keep if lv in H4_CONJUNCT_LEVELS else ())
        shk[lv]["targets"], shk[lv]["clusters"] = spec["target_labels"], spec["clusters"]
    out["t1b_drift"] = {str(k): {m: _strip(v) for m, v in blk.items()} for k, blk in drift.items()}
    out["t1b_drift_shakespeare"] = {str(k): {m: _strip(v) for m, v in blk.items()} for k, blk in shk.items()}
    clusters = drift[0]["clusters"] + shk[0]["clusters"]
    n_b, n_s = len(set(drift[0]["clusters"])), len(set(shk[0]["clusters"]))

    def has(m):
        return all(_shares_of(blk[lv], m) is not None for blk in (drift, shk) for lv in DRIFT_LEVELS)

    def pooled(m, key="per_target_share", lvls=DRIFT_LEVELS):
        return {lv: list(drift[lv][m][key]) + list(shk[lv][m][key]) for lv in lvls}
    # manipulation checks
    lvl0 = {w["id"]: w["sentences"] for s in (dev, Set("t1e_fresh_namefree"), Set("t1d_films_namefree")) for w in s.works}
    man, intro = {}, {}
    for lv in DRIFT_LEVELS[1:]:
        vals = []
        for nm in _drift_sets(lv):
            for w in Set(nm).works:
                o = lvl0[w["id"]]
                vals.append(realised_replacement(o, w["sentences"], set(drift_banned_types(w["id"], o, lv))))
                intro[(lv, w["id"])] = introduced_types(o, w["sentences"])
        man[lv] = {"banned_token_share_removed": float(np.mean([v["banned_token_share_removed"] for v in vals])),
                   "content_type_survival": float(np.mean([v["content_type_survival"] for v in vals])), "n_works": len(vals)}
    surv = [1.0] + [man[lv]["content_type_survival"] for lv in DRIFT_LEVELS[1:]]
    man["min_adjacent_spacing"] = float(min(a - b for a, b in zip(surv[:-1], surv[1:])))
    man["levels_merged"] = man["min_adjacent_spacing"] < 0.15
    out["t1b_manipulation"] = man
    tf = rivals["tfidf"]
    if has(tf):
        out["t1b_tfidf_pooled_share"] = {lv: float(cluster_means(np.asarray(pooled(tf)[lv]), clusters).mean()) for lv in DRIFT_LEVELS}
    # source-leak index: TF-IDF on the introduced words only (rows) against the untouched candidates
    leak = {}
    for lv in H4_LEVELS:
        leak[lv] = _leak_index(lv, intro, dev, plays_nf, n_perm=min(n_perm, 5000))
    out["t1b_source_leak"] = leak
    leaking = sorted({t for lv in leak for t in leak[lv].get("leaking_targets", [])})
    confounded = any(leak[lv].get("pooled_p", 1.0) < 0.05 for lv in leak)
    out["H4"] = {}
    for x in MODELS:
        nm = nmeth[x]
        if not (has(nm) and all(has(rv) for rv in rivals.values())):
            continue
        res = {"by_rival": {}}
        for r, rv in rivals.items():
            res["by_rival"][r] = h4_head_to_head(pooled(rv), pooled(nm), clusters)
        res["p_iut"] = iut([v["p"] for v in res["by_rival"].values()])
        res["stat"] = min(v["stat"] for v in res["by_rival"].values())
        res["falsified"] = any(v["falsification"]["p"] < ALPHA for v in res["by_rival"].values())
        conj = {}
        for lv in H4_CONJUNCT_LEVELS:
            if "null" in drift[lv].get(nm, {}) and "null" in shk[lv].get(nm, {}):
                conj[lv] = pooled_lineage([drift[lv][nm], shk[lv][nm]], [n_b, n_s])
        res["conjunct_neuron_lineage"] = conj
        res["conjunct_ok"] = bool(conj) and all(v["p"] < ALPHA for v in conj.values())
        res["source_leak_confounded"] = confounded
        if confounded and leaking:
            keepi = [i for i, t in enumerate(drift[0]["targets"] + shk[0]["targets"]) if t not in leaking]
            sub = lambda d: {lv: [d[lv][i] for i in keepi] for lv in d}
            cl = [clusters[i] for i in keepi]
            res["without_leaking"] = {r: h4_head_to_head(sub(pooled(rv)), sub(pooled(nm)), cl) for r, rv in rivals.items()}
            res["without_leaking"]["p_iut"] = iut([v["p"] for k, v in res["without_leaking"].items() if k != "p_iut"])
        res["retention"] = {r: retention_contrast(pooled(rv), pooled(nm), pooled(rv, "per_target_null_mean"),
                                                  pooled(nm, "per_target_null_mean"), clusters) for r, rv in rivals.items()}
        res["degradation_descriptive"] = {r: drift_interaction(pooled(rv), pooled(nm), clusters) for r, rv in rivals.items()}
        res["batman_only"] = {r: h4_head_to_head({lv: drift[lv][rv]["per_target_share"] for lv in DRIFT_LEVELS},
                                                 {lv: drift[lv][nm]["per_target_share"] for lv in DRIFT_LEVELS}, drift[0]["clusters"])
                              for r, rv in rivals.items()}
        res["shakespeare_only"] = {r: h4_head_to_head({lv: shk[lv][rv]["per_target_share"] for lv in DRIFT_LEVELS},
                                                      {lv: shk[lv][nm]["per_target_share"] for lv in DRIFT_LEVELS}, shk[0]["clusters"])
                                   for r, rv in rivals.items()}
        flagged = {w["id"] for lv in DRIFT_LEVELS[1:] for nmset in _drift_sets(lv) for w in Set(nmset).works if w.get("drift_flags")}
        if flagged:
            keepi = [i for i, t in enumerate(drift[0]["targets"] + shk[0]["targets"]) if t not in flagged]
            sub = lambda d: {lv: [d[lv][i] for i in keepi] for lv in d}
            res["without_flagged"] = {r: h4_head_to_head(sub(pooled(rv)), sub(pooled(nm)), [clusters[i] for i in keepi])
                                      for r, rv in rivals.items()}
        comp = _components(drift[0]["targets"]) + shk[0]["clusters"]
        res["components_clustering"] = {r: h4_head_to_head(pooled(rv), pooled(nm), comp) for r, rv in rivals.items()}
        if "style" in drift and _shares_of(drift["style"], nm) is not None:
            st = {m_: list(drift["style"][m_]["per_target_share"]) + list(shk["style"][m_]["per_target_share"]) for m_ in (nm, *rivals.values())}
            res["style_control_advantage"] = {r: advantage_test(st[nm], st[rv], clusters) for r, rv in rivals.items()}
        out["H4"][x] = res
    sec = {}
    for rv in methods:
        for x in MODELS:
            nm = nmeth[x]
            if rv != nm and has(rv) and has(nm):
                sec[f"{nm}_vs_{rv}"] = h4_head_to_head(pooled(rv), pooled(nm), clusters)
    pe = resolve(f"emb:{PRIMARY_ENCODER}", sel)
    if has(pe) and has(tf):
        sec["S1_emb_vs_tfidf"] = h4_head_to_head(pooled(tf), pooled(pe), clusters)
    out["t1b_secondary"] = sec
    out["t1b_curves"] = {m: {str(lv): float(cluster_means(np.asarray(pooled(m)[lv]), clusters).mean()) for lv in DRIFT_LEVELS}
                         for m in methods if has(m)}
    out["t1b_survival_curve"] = {str(lv): (1.0 if lv == 0 else man[lv]["content_type_survival"]) for lv in DRIFT_LEVELS}
    # ---- T1c Chinese (machine-translated by agents), language-matched presence bars
    zh = _maybe("t1c_zh")
    if zh is not None:
        spec = assemble_batman_spec("t1c", zh, dev, _maybe("t1c_fresh_zh"), zh, methods, cand_ref_set=dev)
        r = run_lineage(spec, methods, n_perm=n_perm)
        r["contrasts"] = {}
        for x in MODELS:
            for e in (pe, resolve("emb:e5-base", sel), resolve("emb:labse", sel), resolve(f"llmsum:{x}", sel)):
                if _shares_of(r, nmeth[x]) and _shares_of(r, e):
                    r["contrasts"][f"{nmeth[x]}_vs_{e}"] = method_contrast(r[nmeth[x]]["per_target_share"], r[e]["per_target_share"],
                                                                           spec["clusters"], alternative="two-sided")
        out["t1c_zh"] = {m: _strip(v) if isinstance(v, dict) and "stat" in v else v for m, v in r.items()}
        mt = _maybe("t1c_zh_mt")
        if mt is not None:
            lex = [resolve(k, sel) for k in ("tfidf", "words", "lsa")]
            spec = assemble_batman_spec("t1c_mt", mt, dev, _maybe("t1c_fresh_zh_mt"), mt, lex)
            out["t1c_zh_mt_words"] = {m: _strip(v) for m, v in run_lineage(spec, lex, n_perm=n_perm, residual=False).items()}
        nat = _maybe("t1c_zhwiki")
        if nat is not None:
            try:
                spec = assemble_batman_spec("t1c_natural", nat, dev, None, zh, methods, cand_ref_set=dev)
                out["t1c_natural"] = {m: _strip(v) for m, v in run_lineage(spec, methods, n_perm=n_perm, residual=False).items()}
                out["t1c_natural"]["targets"] = spec["target_labels"]
            except (KeyError, ValueError) as err:
                out["t1c_natural"] = {"error": str(err)}
        zcc, dcc = _maybe("t1c_zh_cc"), _maybe("dev_namefree_cc")
        if zcc is not None and dcc is not None:
            spec = assemble_batman_spec("t1c_cc", zcc, dcc, _maybe("t1c_fresh_zh_cc"), zcc, methods, cand_ref_set=dcc)
            out["t1c_codes_collapsed"] = {m: _strip(v) for m, v in run_lineage(spec, methods, n_perm=n_perm, residual=False).items()}
    # ---- T1d Shakespeare (H5)
    plays, films = Set("t1d_plays_namefree"), Set("t1d_films_namefree")
    titles = {w["id"]: w["id"] for w in films.works}
    rec = _recognised_ids(films, titles=titles) or set()
    single = {p for p, c in Counter(w["sources"][0] for w in films.works if w.get("h5")).items() if c == 1}
    subsets = {"h5": lambda w: w.get("h5"), "renamed_all_evidence": lambda w: w.get("renamed"),
               "h5_without_single_film_plays": lambda w: w.get("h5") and w["sources"][0] not in single,
               "h5_unrecognised": lambda w: w.get("h5") and w["id"] not in rec,
               "faithful": lambda w: not w.get("renamed"), "all_primary": lambda w: True}
    out["t1d"] = {}
    for sub, filt in subsets.items():
        spec = assemble_shakespeare_spec(f"t1d_{sub}", films, plays, dev, methods, filt)
        if not spec["targets"]:
            continue
        r = run_lineage(spec, methods, n_perm=n_perm, residual=sub == "h5")
        r["contrasts"] = {}
        for x in MODELS:
            for rv in methods:
                if rv != nmeth[x] and _shares_of(r, nmeth[x]) and _shares_of(r, rv):
                    r["contrasts"][f"{nmeth[x]}_vs_{rv}"] = advantage_test(r[nmeth[x]]["per_target_share"], r[rv]["per_target_share"],
                                                                           spec["clusters"], SESOI["H5"] if rv in rivals.values() else None,
                                                                           key=f"t1d{sub}{rv}")
        out["t1d"][sub] = {m: _strip(v) if isinstance(v, dict) and "stat" in v else v for m, v in r.items()}
    h5 = out["t1d"].get("h5", {})
    out["H5"] = {}
    for x in MODELS:
        by = {r: h5.get("contrasts", {}).get(f"{nmeth[x]}_vs_{rv}") for r, rv in rivals.items()}
        if all(by.values()):
            out["H5"][x] = {"by_rival": by, "p_iut": iut([v["p"] for v in by.values()]), "stat": min(v["stat"] for v in by.values()),
                            "falsified": any(v["falsification"]["p"] < ALPHA for v in by.values())}
    inform = [m for m, v in h5.items() if isinstance(v, dict) and "p" in v and "|" not in m and v["p"] < ALPHA]
    out["t1d_informative_methods"] = inform
    cc_p, cc_f = _maybe("t1d_plays_namefree_cc"), _maybe("t1d_films_namefree_cc")
    if cc_p is not None and cc_f is not None:
        spec = assemble_shakespeare_spec("t1d_h5_cc", cc_f, cc_p, dev, methods, subsets["h5"])
        out["t1d_codes_collapsed"] = {m: _strip(v) for m, v in run_lineage(spec, methods, n_perm=n_perm, residual=False).items()}
    plays_o, films_o = _maybe("t1d_plays_original"), _maybe("t1d_films_original")
    if plays_o is not None and films_o is not None:
        spec = assemble_shakespeare_spec("t1d_control", films_o, plays_o, Set("dev_original"), [resolve("tfidf")],
                                         lambda w: w.get("names_flag") == "keeps_names")
        out["t1d_positive_control"] = {m: _strip(v) for m, v in run_lineage(spec, [resolve("tfidf")], n_perm=2000, residual=False).items()}
    # ---- T1e fresh adaptations, with the 8-gram reuse audit
    fresh = Set("t1e_fresh_namefree")
    spec = assemble_batman_spec("t1e", dev, dev, fresh, dev, methods, fresh_only=True)
    out["t1e_fresh"] = {m: _strip(v) for m, v in run_lineage(spec, methods, n_perm=n_perm).items()}
    out["t1e_fresh"]["targets"] = spec["target_labels"]
    out["reuse_audit"] = _reuse_audit(dev, fresh)
    out["reuse_sensitivity"] = _reuse_sensitivity(out["reuse_audit"], dev, fresh, methods, nmeth, rivals, n_perm)
    # ---- the codes-only baseline is part of every block ('codes'); BH over the T1 secondaries
    pv = {}
    for key, block in (("drift100", drift[100]), ("shkdrift100", shk[100]), ("zh", out.get("t1c_zh", {})),
                       ("shk", out["t1d"].get("h5", {})), ("fresh", out["t1e_fresh"])):
        for m, v in block.items():
            if isinstance(v, dict) and "p" in v and "stat" in v:
                pv[f"{key}:{m}"] = v["p"]
    for k, v in sec.items():
        pv[f"t1b:{k}"] = v["p"]
    out["t1_bh"] = bh(pv)
    out["analysis_timing"] = {k: dict(v) for k, v in TIMING.items()}
    return out


def _strip(v):
    if isinstance(v, dict):
        return {k: x for k, x in v.items() if k != "null"}
    return v


def _components(batman_targets):
    """Connected components of the Batman target-source graph (sensitivity clustering for H4)."""
    src = {t: s for t, s in BAT_ADAPTATIONS.items()}
    src.update({t: v[1] for t, v in FRESH_ADAPTATIONS.items()})
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            a = parent[a]
        return a
    for t in batman_targets:
        for s in src[t]:
            parent[find(t)] = find("comic:" + s)
    return [f"component:{find(t)}" for t in batman_targets]


def _leak_index(lv, intro, dev, plays_nf, n_perm=5000):
    """Source-leak index at one drift level: TF-IDF lineage test where each drifted row is represented only by the
    words its rewrite introduced. Leaking targets: their documented source ranks first (share 0)."""
    a, b, c = _drift_sets(lv)
    out = {"by_arm": {}}
    res = []
    for arm, rows_set, cands, cand_idx, kind in (("batman", (Set(a), Set(b)), dev, _comics(dev), "batman"),
                                                 ("shakespeare", (Set(c),), plays_nf, list(range(len(plays_nf.ids))), "shk")):
        rows = [(S, i) for S in rows_set for i in range(len(S.ids))]
        if kind == "batman":
            rows = [(S, i) for S, i in rows if S.works[i]["id"] in BAT_ADAPTATIONS or S.works[i]["id"] in FRESH_ADAPTATIONS
                    or S.works[i]["id"] in BAT_SCREEN_NONADAPT]
        else:
            rows = [(S, i) for S, i in rows if S.works[i].get("drift_sample")]
        texts = [" ".join(sorted(intro.get((lv, S.works[i]["id"]), set()))) or "empty" for S, i in rows]
        ctexts = [_texts(cands)[j] for j in cand_idx]
        X, _ = tfidf_matrix(texts + ctexts, texts + ctexts, min_df=1)
        Sim = cosine(X[:len(texts)].toarray(), X[len(texts):].toarray())
        ids = [S.works[i]["id"] for S, i in rows]
        if kind == "batman":
            cid = [cands.ids[j] for j in cand_idx]
            cyear = np.array([cands.works[j]["year"] for j in cand_idx])
            tg = [k for k, t in enumerate(ids) if t in BAT_ADAPTATIONS or t in FRESH_ADAPTATIONS]
            srcs = [[cid.index(s) for s in (BAT_ADAPTATIONS.get(ids[k]) or FRESH_ADAPTATIONS[ids[k]][1])] for k in tg]
            yrs = [FRESH_ADAPTATIONS[ids[k]][0] if ids[k] in FRESH_ADAPTATIONS else dev.works[dev.index[ids[k]]]["year"] for k in tg]
            el = np.array([[cy <= y for cy in cyear] for y in yrs])
            hub = hub_groups_fixed(len(ids), [k for k, t in enumerate(ids) if t in BAT_SCREEN_NONADAPT], len(cand_idx))
            r = lineage_test(Sim, tg, srcs, el, hub, n_perm=n_perm, keep_null=True, rng=rng_for("leak", lv, arm))
        else:
            srcs_all = [[plays_nf.ids.index(p) for p in S.works[i]["sources"]] for S, i in rows]
            tg = list(range(len(rows)))
            play_of = [S.works[i]["sources"][0] for S, i in rows]
            cnt = Counter(play_of)
            r = lineage_test(Sim, tg, srcs_all, np.ones((len(tg), len(cand_idx)), bool),
                             hub_groups_other_targets(len(rows), tg, srcs_all, len(cand_idx)),
                             weights=[1.0 / (len(cnt) * cnt[p]) for p in play_of], n_perm=n_perm, keep_null=True, rng=rng_for("leak", lv, arm))
        r["leaking_targets"] = [ids[t] for t, s in zip(tg, r["per_target_share"]) if s == 0]
        res.append(r)
        out["by_arm"][arm] = _strip(r)
    out["pooled_p"] = pooled_lineage(res, [1, 1])["p"]
    out["leaking_targets"] = sorted({t for r in res for t in r["leaking_targets"]})
    return out


def _reuse_audit(dev, fresh):
    """Shared word 8-grams between each Batman target and its documented source and the other comics (T1e and level
    0); the sentences that share one are listed for the reuse sensitivity."""
    comics = {dev.ids[i]: dev.works[i]["sentences"] for i in _comics(dev)}
    out = {}
    for S, targets in ((dev, list(BAT_ADAPTATIONS)), (fresh, list(FRESH_ADAPTATIONS))):
        for t in targets:
            if t not in S.index:
                continue
            sents = S.works[S.index[t]]["sentences"]
            srcs = BAT_ADAPTATIONS.get(t) or FRESH_ADAPTATIONS[t][1]
            out[t] = {"source_shared_sentences": sorted({i for s in srcs for i in eightgram_overlap(sents, comics[s])}),
                      "other_shared_sentences": sorted({i for c, cs in comics.items() if c not in srcs for i in eightgram_overlap(sents, cs)})}
    return out


class MaskedSet(Set):
    """A copy of a set with some sentences removed (reuse sensitivity): texts drop them, and neuron aggregates are
    recomputed from full.npy (top-3 over the remaining sentences) when the set has full activations."""

    def __init__(self, base, drop):
        self.__dict__.update(base.__dict__)
        self.base, self.drop = base, drop
        keep = [[s for k, s in enumerate(w["sentences"]) if k not in drop.get(w["id"], ())] or w["sentences"][:1] for w in base.works]
        self.works = [dict(w, sentences=k) for w, k in zip(base.works, keep)]
        self.name = base.name + "_reusefree"

    def texts(self):
        return [lexical_text(w["sentences"]) for w in self.works]

    def agg(self, model):
        F = self.base.full(model)
        A_ = np.array(self.base.agg(model), np.float32)
        for wi, w in enumerate(self.base.works):
            d = self.drop.get(w["id"])
            if not d:
                continue
            rows = np.where(self.base.sent_work == wi)[0]
            keep = [r for k, r in enumerate(rows) if k not in d] or list(rows[:1])
            A_[wi] = topk_mean(np.asarray(F[keep], np.float32))
        return A_


def _reuse_sensitivity(audit, dev, fresh, methods, nmeth, rivals, n_perm):
    """Level-0 Batman and T1e without the sentences that share an 8-gram with any comic (text methods always;
    neurons when the sets have full activations)."""
    drop = {t: set(v["source_shared_sentences"]) | set(v["other_shared_sentences"]) for t, v in audit.items()}
    if not any(drop.values()):
        return {"note": "no shared 8-grams"}
    out = {}
    for label, S_rows, S_fresh, fo in (("level0", MaskedSet(dev, drop), None, False), ("t1e", dev, MaskedSet(fresh, drop), True)):
        ms = [m for m in methods if _family(m)[1] in ("tfidf", "lsa", "words")]
        ok_full = all((S_.has(f"neurons_{x}/full.npy") if hasattr(S_, "has") else False)
                      for S_ in ((S_rows.base if isinstance(S_rows, MaskedSet) else S_rows),) for x in MODELS)
        if label == "t1e":
            ok_full = all(fresh.has(f"neurons_{x}/full.npy") for x in MODELS)
        if ok_full:
            ms += list(nmeth.values())
        try:
            spec = assemble_batman_spec(f"reuse_{label}", S_rows, dev, S_fresh if S_fresh is not None else (fresh if not fo else None),
                                        dev, ms, fresh_only=fo)
            r = run_lineage(spec, ms, n_perm=min(n_perm, 5000), residual=False)
            out[label] = {m: _strip(v) for m, v in r.items()}
            for x in MODELS:
                if nmeth[x] in r and "per_target_share" in r[nmeth[x]]:
                    for rk, rv in rivals.items():
                        if rv in r and "per_target_share" in r[rv]:
                            out[label][f"{nmeth[x]}_vs_{rv}"] = advantage_test(r[nmeth[x]]["per_target_share"], r[rv]["per_target_share"],
                                                                               spec["clusters"], key=f"reuse{label}")
        except (FileNotFoundError, KeyError) as err:
            out[label] = {"error": str(err)}
    return out


# =============================================================================================== T2: planted memes
JUDGE_TASKS = (("judge", "judge"), ("judge-shared", "judge-shared"), ("judge-1pass", "judge-1pass"), ("sjudge", "summary-judge"))


def _neuron_cols(S, model):
    if S.has(f"neurons_{model}/sent.npy"):
        sv, units = S.sent_units(model)
    elif S.has(f"neurons_{model}/full.npy"):
        sv = S.full(model)
        units = np.arange(sv.shape[1])
    else:
        return None, None
    return sv, {int(u): i for i, u in enumerate(units)}


def _det_scores(sv, col, det, sent_work, n, length_fix=None, wc=None):
    us = det["units"]
    X = np.stack([np.asarray(sv[:, col[u]], np.float32) for u in us], 1)
    if length_fix is not None:                      # length-residualised sensitivity: x - slope_u * log(words)
        X = X - np.array([length_fix[str(u)] for u in us])[None, :] * np.log1p(wc)[:, None]
    Xs = ((X - np.array(det["q50"])) / np.array(det["scale"])).mean(1)
    return work_from_sentences(Xs, sent_work, n)


def meme_story_scores(S, memes, freeze, set_key="main", length_resid=False, only_neurons=False):
    """Story-by-meme score matrices for every T2 method on one set (PREREGISTRATION.md section 2 and T2). Every
    detector was fitted at F1 on the same inputs for every method (definition + example set) against the same
    contrastive background. Missing features are skipped (and reported as missing by the caller)."""
    ids = [m["id"] for m in memes]
    n = len(S.ids)
    out = {}
    t2 = freeze.get("t2", {})
    wc = np.array([len(s.split()) for s in S.sentences], float)
    for model in MODELS:
        sv, col = _neuron_cols(S, model)
        if sv is None:
            continue
        det_all = t2.get("neurons", {}).get(model, {})
        chosen = t2.get("neurons_selected", {}).get(model, "m1")
        fix = t2.get("length_slopes", {}).get(model) if length_resid else None
        for st in ("main", "explicit", "far"):
            if st not in det_all:
                continue
            for variant in ("m1", "m5"):
                M = np.full((n, len(ids)), np.nan)
                for j, mid in enumerate(ids):
                    d = det_all[st].get(mid, {}).get(variant)
                    if d and all(u in col for u in d["units"]):
                        M[:, j] = _det_scores(sv, col, d, S.sent_work, n, fix, wc)
                if st == "main":
                    out[f"neurons:{model}:{variant}"] = M
                    if variant == chosen:
                        out[f"neurons:{model}"] = M
                elif variant == chosen:
                    out[f"neurons:{model}:{st}"] = M
        ctx = t2.get("neurons_ctx", {}).get(model, {})
        if ctx:
            M = np.full((n, len(ids)), np.nan)
            for j, mid in enumerate(ids):
                d = ctx.get(mid)
                if d and all(u in col for u in d["units"]):
                    M[:, j] = _det_scores(sv, col, d, S.sent_work, n)
            out[f"neurons:{model}:m1ctx"] = M
        M = np.full((n, len(ids)), np.nan)
        for j, mid in enumerate(ids):
            u = t2.get("devunits", {}).get(model, {}).get(mid)
            if u is not None and int(u) in col:
                M[:, j] = work_from_sentences(np.asarray(sv[:, col[int(u)]], np.float32), S.sent_work, n)
        out[f"devunit:{model}"] = M
    if only_neurons:
        return out
    t0 = time.time()
    cache = StemCache(S.sentences)
    for src, q in t2.get("keywords", {}).items():
        M = np.zeros((n, len(ids)))
        for j, mid in enumerate(ids):
            terms = q.get(mid, [])
            hits = np.array([cache.hits(i, terms) for i in range(len(S.sentences))], float)
            tot = np.bincount(S.sent_work, weights=hits, minlength=n)
            M[:, j] = work_from_sentences(hits, S.sent_work, n) + 1e-3 * tot
        out[f"keywords:{src}"] = M
    lr = t2.get("logreg")
    if lr:
        vec = lexical_vectorizer_from(lr["vocab"], lr["idf"])
        X = vec.transform([lexical_text([s]) for s in S.sentences])
        for st in ("main", "far"):
            if st not in lr["det"]:
                continue
            M = np.full((n, len(ids)), np.nan)
            for j, mid in enumerate(ids):
                d = lr["det"][st].get(mid)
                if d:
                    M[:, j] = work_from_sentences(np.asarray(X @ np.asarray(d["coef"], np.float64)).ravel() + d["b"], S.sent_work, n)
            out["keywords:logreg" + ("" if st == "main" else ":far")] = M
    TIMING[S.name]["keywords and lexical probe (scoring)"] = time.time() - t0
    for enc, by in t2.get("emb", {}).items():
        kind = t2.get("emb_selected", {}).get(enc, "probe")
        for unit, prefix in (("sentence", "emb"), ("win3", "embwin"), ("doc", "embdoc")):
            suffix = {"sentence": "", "win3": "_win3", "doc": "_doc"}[unit]
            if not S.has(f"emb_{enc}{suffix}.npy"):
                continue
            E = np.asarray(S.emb(enc, unit), np.float32)
            for st in ("main", "far"):
                if st not in by or (st == "far" and unit != "sentence"):
                    continue
                M = np.full((n, len(ids)), np.nan)
                for j, mid in enumerate(ids):
                    d = by[st].get(mid, {}).get(kind)
                    if d:
                        sc = score_linear(E, d)
                        M[:, j] = sc if unit == "doc" else work_from_sentences(sc, S.sent_work, n)
                out[f"{prefix}:{enc}" + ("" if st == "main" else ":far")] = M
    for model in MODELS:
        h = t2.get("hidden", {}).get(model)
        if not h or not S.has(f"neurons_{model}/hidden.npy"):
            continue
        Hs = (np.asarray(S.hidden(model)[:, h["layer"]], np.float32) - np.array(h["mu"])) / np.array(h["sd"])
        kind = h.get("selected", "probe")
        for st in ("main", "far"):
            if st not in h["det"]:
                continue
            M = np.full((n, len(ids)), np.nan)
            for j, mid in enumerate(ids):
                d = h["det"][st].get(mid, {}).get(kind)
                if d:
                    M[:, j] = work_from_sentences(score_linear(Hs, d), S.sent_work, n)
            out[f"hidden:{model}" + ("" if st == "main" else ":far")] = M
    for r in list(MODELS) + [SMALL_READER, "claude"]:
        for task, key in JUDGE_TASKS:
            rows = S.llm(r, task)
            if not rows:
                continue
            M = np.full((n, len(ids)), np.nan)
            for x in rows:
                if x["work"] in S.index and x["meme"] in ids:
                    M[S.index[x["work"]], ids.index(x["meme"])] = x["p_yes"]
            name = "judge:claude-upper-bound" if r == "claude" else f"{key}:{r}"
            out[name] = M
    N_ = S.nli()
    if N_ is not None and N_.shape[0] == len(S.sentences):
        nli_ids = t2.get("nli_memes", ids)
        cols = [nli_ids.index(m) if m in nli_ids else None for m in ids]
        M = np.full((n, len(ids)), np.nan)
        for j, c in enumerate(cols):
            if c is not None:
                M[:, j] = work_from_sentences(np.asarray(N_[:, c], np.float32), S.sent_work, n)
        out["nli:deberta-v3-base"] = M
    out["length"] = np.repeat(work_from_sentences(wc, S.sent_work, n)[:, None], len(ids), 1)
    return out


def topic_story_scores(S, memes, freeze, background_texts, example_texts, encoder=PRIMARY_ENCODER):
    """Topic rivals. LDA (K chosen at F1 between 50 and 100) fitted on background + examples + story sentences (no
    labels); the meme's topic = argmax of mean example theta minus mean contrastive-background theta; sentence score =
    that topic's theta. Embedding clusters: HDBSCAN on the reduced embeddings of the same sentences; the meme's
    cluster = the cluster nearest most of its examples; sentence score = minus the distance to its centroid."""
    ids = [m["id"] for m in memes]
    n = len(S.ids)
    out = {}
    K = freeze.get("t2", {}).get("lda_K", 50)
    ex_all = [e for m in memes for e in example_texts[m["id"]]]
    fit = list(background_texts) + ex_all + list(S.sentences)
    t0 = time.time()
    th, _ = lda_fit_transform(fit, fit, K=K, seed=0, max_iter=20)
    TIMING[S.name][f"lda K={K} fit and score"] = time.time() - t0
    nb, ne = len(background_texts), len(ex_all)
    thb, the, ths = th[:nb], th[nb:nb + ne], th[nb + ne:]
    M = np.zeros((n, len(ids)))
    k0 = 0
    for j, mid in enumerate(ids):
        k = len(example_texts[mid])
        own = np.zeros(len(the), bool)
        own[k0:k0 + k] = True
        t = int(np.argmax(the[own].mean(0) - np.concatenate([thb, the[~own]]).mean(0)))
        k0 += k
        M[:, j] = work_from_sentences(ths[:, t], S.sent_work, n)
    out["lda"] = M
    Bset, Xset = _maybe("t2_background"), _maybe("t2_examples")
    if S.has(f"emb_{encoder}.npy") and Bset is not None and Bset.has(f"emb_{encoder}.npy") and Xset is not None and Xset.has(f"emb_{encoder}.npy"):
        Eb = np.asarray(Bset.emb(encoder), np.float32)
        Ee_all = np.asarray(Xset.emb(encoder), np.float32)
        rows = [i for i, w in enumerate(Xset.works) if w.get("set") in EXAMPLE_SETS["main"]]
        Ee = Ee_all[rows]
        owner = [Xset.works[i]["meme"] for i in rows]
        Es = np.asarray(S.emb(encoder), np.float32)
        lab, cents, R, how, _ = cluster_topics(np.concatenate([Eb, Ee, Es]))
        Re, Rs = R[len(Eb):len(Eb) + len(Ee)], R[len(Eb) + len(Ee):]
        M = np.full((n, len(ids)), np.nan)
        for j, mid in enumerate(ids):
            sel = [k for k, o in enumerate(owner) if o == mid]
            if len(cents) and sel:
                near = np.argmin(((Re[sel][:, None, :] - cents[None]) ** 2).sum(-1), 1)
                c = cents[Counter(near.tolist()).most_common(1)[0][0]]
                M[:, j] = work_from_sentences(-np.sqrt(((Rs - c) ** 2).sum(1)), S.sent_work, n)
        out[f"topics:{encoder}"] = M
        out["_reducer"] = how
    return out


def _design(S):
    return [{"plot": w["plot"], "version": w["version"], "memes": w["memes"], "exclude_memes": w.get("exclude_memes", []),
             "dropped_memes": w.get("dropped_memes", []), "writer": w.get("writer"), "genre": w.get("genre")} for w in S.works]


def _reads_kappa(S, memes):
    """A1: Cohen's kappa between the two Claude readers, and between the first Claude reader and the cross-family
    reader, over story-by-meme cells, per version."""
    out = {}
    for v in VERSIONS:
        a, b, c, x = [], [], [], []
        for w in S.works:
            rd = [set(r) for r in (w.get("reads_claude") or [])]
            ot = [set(r) for r in (w.get("reads_other") or [])]
            if w["version"] != v or len(rd) < 2:
                continue
            for m in memes:
                a.append(m in rd[0])
                b.append(m in rd[1])
                if ot:
                    c.append(m in rd[0])
                    x.append(m in ot[0])
        out[v] = {"claude_claude": cohen_kappa(a, b) if a else None, "claude_cross_family": cohen_kappa(c, x) if c else None}
    return out


def _human_audit(S, memes):
    """The author's blind audit (annotation_protocol.md A7): presence agreement on reworded and coded planted cells,
    absence agreement on control and decoy cells, and human-Claude kappa."""
    p = DATA / "human_audit.jsonl"
    if not p.exists():
        return None
    rows = read_jsonl(p)
    t2 = [r for r in rows if r.get("kind") == "t2_cell"]
    idx = {w["id"]: w for w in S.works}
    pres, absn, hum, cla = [], [], [], []
    for r in t2:
        w = idx.get(r["story"])
        if w is None:
            continue
        planted = r["meme"] in w["memes"] and w["version"] in PLANTED
        if planted and w["version"] in REWORDED | {"coded"}:
            pres.append(bool(r["present"]))
        if r["meme"] in w["memes"] and w["version"] in ("control", "decoy"):
            absn.append(not r["present"])
        rd = [set(x) for x in (w.get("reads_claude") or [])]
        if rd:
            hum.append(bool(r["present"]))
            cla.append(all(r["meme"] in x for x in rd))
    return {"n_cells": len(t2), "presence_agreement_reworded": float(np.mean(pres)) if pres else None,
            "absence_agreement_controls": float(np.mean(absn)) if absn else None,
            "kappa_human_claude": cohen_kappa(hum, cla) if hum else None,
            "pass": bool(pres and absn and np.mean(pres) >= 0.85 and np.mean(absn) >= 0.95),
            "other_items": Counter(r.get("kind") for r in rows if r.get("kind") != "t2_cell")}


def run_t2(freeze, memes_file=PRE / "memes_for_tracing.json"):
    """T2: planted memes (H1-H3) plus Chinese, natural-text and long-text secondaries."""
    memes = json.load(open(memes_file))["memes"]
    ids = [m["id"] for m in memes]
    tier = {m["id"]: m["tier"] for m in memes}
    S = Set("t2_stories")
    design = _design(S)
    valid = np.array([w.get("valid", True) for w in S.works])
    dropped = sorted({m for w in S.works for m in w.get("dropped_memes", [])})
    use = [m for m in ids if m not in dropped]
    scores = meme_story_scores(S, memes, freeze)
    ex = {m["id"]: [m["definition"]] + m["examples"].get("explicit", []) + m["examples"].get("reworded", []) for m in memes}
    tops = topic_story_scores(S, memes, freeze, Set("t2_background").sentences, ex)
    reducer = tops.pop("_reducer", None)
    scores.update(tops)
    if freeze.get("t2", {}).get("length_slopes"):
        lr_ = meme_story_scores(S, memes, freeze, length_resid=True, only_neurons=True)
        for model in MODELS:
            if f"neurons:{model}" in lr_:
                scores[f"length_resid:{model}"] = lr_[f"neurons:{model}"]
    ref = _maybe("t2_reference")
    ref_scores = meme_story_scores(ref, memes, freeze) if ref is not None else {}
    out = {"version": VERSION, "n_stories": len(S.ids), "n_valid": int(valid.sum()), "dropped_memes": dropped, "topic_reducer": reducer,
           "by_method": {}, "exclusions": _exclusion_counts(S), "reader_kappa": _reads_kappa(S, ids), "human_audit": _human_audit(S, ids),
           "cross_family_check": _xfamily_check(S, ids), "salience": _salience_summary(S)}
    jcols = {i: ids.index(i) for i in use}
    for name, M in scores.items():
        Mu = M[:, [jcols[m] for m in use]]
        r = {c: detection_aucs(Mu, design, use, cs, valid, "twin") for c, cs in CONDITIONS.items()}
        r["itt_reworded"] = detection_aucs(Mu, design, use, REWORDED, valid, "twin", itt=True)
        for sch in ("matched", "all", "hard", "vs_decoy"):
            r[f"{sch}_reworded"] = detection_aucs(Mu, design, use, REWORDED, valid, sch)
        r["decoy_vs_control"] = detection_aucs(Mu, design, use, {"decoy"}, valid, "decoy")
        r["r_precision_reworded"] = precision_at_r(Mu, design, use, REWORDED, valid)
        barv = np.full(len(use), 0.5) if name.startswith(("judge", "summary-judge", "nli")) else (
            presence_bar(ref_scores[name][:, [jcols[m] for m in use]]) if name in ref_scores else None)
        if barv is not None:
            r["f1_at_bar_reworded"] = f1_at_bar(Mu, design, use, REWORDED, barv, valid)
        r["mean"] = {k: float(np.nanmean(list(v.values()))) if v else None for k, v in r.items() if isinstance(v, dict)}
        r["by_writer_reworded"] = {wr: float(np.nanmean(list(detection_aucs(Mu, [d if d["writer"] == wr else dict(d, version="_") for d in design],
                                                                            use, REWORDED, valid, "twin").values())))
                                   for wr in sorted({d["writer"] for d in design if d["writer"]})}
        r["by_tier_reworded"] = {t: float(np.nanmean([r["reworded"][m] for m in use if tier[m] == t])) for t in sorted(set(tier.values()))}
        out["by_method"][name] = r
    bym = out["by_method"]
    out["H1"], out["H2"], out["H3"] = {}, {}, {}
    neutral = [m for m in use if tier[m] == "N"]
    for model in MODELS:
        nk = f"neurons:{model}"
        if nk not in bym:
            continue
        h1 = {"by_rival": {}}
        for rv in H1_RIVALS:
            if rv in bym:
                t = _t2_compare(scores, bym, nk, rv, use, jcols, design, valid, 0.0, SESOI["H1"])
                t["neutral_subset_mean_d"] = float(np.nanmean([t["per_meme_d"][m] for m in neutral])) if neutral else None
                t["itt"] = paired_meme_test(bym[nk]["itt_reworded"], bym[rv]["itt_reworded"], use, "greater")
                t["two_way_ci"] = two_way_bootstrap(scores[nk][:, [jcols[m] for m in use]], scores[rv][:, [jcols[m] for m in use]],
                                                    design, use, REWORDED, valid, key=f"h1{model}{rv}")
                t["leave_one_writer_out"] = _lowo(scores[nk], scores[rv], design, use, jcols, valid)
                h1["by_rival"][rv] = t
        if len(h1["by_rival"]) == len(H1_RIVALS):
            h1["p_iut"] = iut([v["p"] for v in h1["by_rival"].values()])
            h1["stat"] = min(v["stat"] for v in h1["by_rival"].values())
            h1["falsified"] = any(v["falsification"]["p"] < ALPHA for v in h1["by_rival"].values())
            h1["neutral_consistent"] = all((v["neutral_subset_mean_d"] or 0) > 0 for v in h1["by_rival"].values())
            h1["itt_p_iut"] = iut([v["itt"]["p"] for v in h1["by_rival"].values()])
        out["H1"][model] = h1
        h2 = {"by_rival": {}}
        for rv in H2_RIVALS:
            if rv in bym:
                ni = _t2_compare(scores, bym, nk, rv, use, jcols, design, valid, -NI_MARGIN, -NI_MARGIN)
                sup = _t2_compare(scores, bym, nk, rv, use, jcols, design, valid, 0.0, SESOI["H2"])
                ni["neutral_subset_mean_d"] = float(np.nanmean([ni["per_meme_d"][m] for m in neutral])) if neutral else None
                h2["by_rival"][rv] = {"noninferiority": ni, "superiority": sup}
        if len(h2["by_rival"]) == len(H2_RIVALS):
            h2["p_iut"] = iut([v["noninferiority"]["p"] for v in h2["by_rival"].values()])
            h2["superiority_p_iut"] = iut([v["superiority"]["p"] for v in h2["by_rival"].values()])
            h2["stat"] = min(v["noninferiority"]["stat"] for v in h2["by_rival"].values())
            h2["falsified"] = any(v["noninferiority"]["falsification"]["p"] < ALPHA for v in h2["by_rival"].values())
            h2["neutral_consistent"] = all((v["noninferiority"]["neutral_subset_mean_d"] or 0) > -NI_MARGIN for v in h2["by_rival"].values())
        out["H2"][model] = h2
        dk = f"devunit:{model}"
        mapped = [m for m in use if freeze.get("t2", {}).get("devunits", {}).get(model, {}).get(m) is not None]
        dclass = freeze.get("t2", {}).get("dev_class", {})
        concept = [m for m in mapped if dclass.get(m) == "concept"]
        if dk in scores and concept:
            Mdet = scores[dk][:, [ids.index(m) for m in concept]]
            A = specificity_matrix(Mdet, concept, design, use, REWORDED, valid)
            h3 = specificity_test(A, concept, use, sesoi=SESOI["H3"])
            h3["falsified"] = h3["falsification"]["p"] < ALPHA
            Mall = scores[dk][:, [ids.index(m) for m in mapped]]
            h3["all_dev_units"] = _strip(specificity_test(specificity_matrix(Mall, mapped, design, use, REWORDED, valid), mapped, use))
            other = [m for m in mapped if m not in concept]
            h3["concept_minus_name_word"] = float(np.nanmean([h3["all_dev_units"]["per_unit_specificity"][m] for m in concept]) -
                                                  np.nanmean([h3["all_dev_units"]["per_unit_specificity"][m] for m in other])) if other else None
            h3["auc_vs_half"] = signflip([bym[dk]["reworded"][m] for m in concept], alternative="greater", null_value=0.5)
            surv = freeze.get("t2", {}).get("dev_survivor", {})
            h3["by_dev_rewording_survivor"] = {str(k): float(np.nanmean([h3["per_unit_specificity"][m] for m in concept if bool(surv.get(m)) == k] or [np.nan]))
                                               for k in (True, False)}
            h3["concept_units"] = concept
            out["H3"][model] = h3
    out.update(_t2_secondaries(bym, scores, design, use, ids, jcols, valid, tier))
    out["secondary_bh"] = bh({k: v.get("p") for k, v in out["secondary"].items() if isinstance(v, dict) and v.get("p") is not None})
    out["zh"] = _t2_zh(memes, freeze)
    out["natural"] = _t2_natural(memes, freeze)
    out["long"] = _t2_long(memes, freeze)
    out["analysis_timing"] = dict(TIMING[S.name])
    return out


def _t2_compare(scores, bym, a, b, use, jcols, design, valid, null_value, sesoi):
    """A primary T2 comparison of method a with method b on reworded twin pairs: the meme-level sign flip and the
    plot-level sign flip must both reject (p = the larger), and 'falsified' needs both falsification tests to reject."""
    t = paired_meme_test(bym[a]["reworded"], bym[b]["reworded"], use, "greater", null_value=null_value, sesoi=sesoi)
    cols = [jcols[m] for m in use]
    Ma, Mb = scores[a][:, cols], scores[b][:, cols]
    pf = plot_flip_test(Ma, Mb, design, use, REWORDED, valid, "greater", null_value=null_value, key=f"pf{a}{b}{null_value}")
    pff = plot_flip_test(Ma, Mb, design, use, REWORDED, valid, "less", null_value=sesoi, key=f"pff{a}{b}{sesoi}")
    t["p_meme_level"], t["plot_level"] = t["p"], pf
    t["p"] = max(t["p"], pf["p"])
    t["falsification"] = {"p_meme_level": t["falsification"]["p"], "p_plot_level": pff["p"], "p": max(t["falsification"]["p"], pff["p"])}
    return t


def _xfamily_check(S, memes):
    """Agreement of the cross-family check reader with the design: presence on banned, implicit and coded planted
    cells, absence on control and decoy cells (the fallback pass rule when there is no human audit)."""
    pres, absn = [], []
    for w in S.works:
        rd = w.get("reads_other") or []
        if not rd:
            continue
        f = set(rd[0])
        for m in w["memes"]:
            if w["version"] in REWORDED | {"coded"}:
                pres.append(m in f)
            elif w["version"] in ("control", "decoy"):
                absn.append(m not in f)
    if not pres:
        return None
    return {"presence_agreement_reworded": float(np.mean(pres)), "absence_agreement_controls": float(np.mean(absn)) if absn else None,
            "pass": bool(np.mean(pres) >= 0.75 and (not absn or np.mean(absn) >= 0.90))}


def _salience_summary(S):
    """Q6: mean slot rating of the banned and implicit versions minus the control version, per plot and overall."""
    by = defaultdict(dict)
    for w in S.works:
        sal = w.get("salience") or {}
        vals = [np.mean(v) for v in sal.values() if v]
        if vals:
            by[w["plot"]][w["version"]] = float(np.mean(vals))
    d = [np.mean([v[x] for x in ("banned", "implicit") if x in v]) - v["control"] for v in by.values()
         if "control" in v and any(x in v for x in ("banned", "implicit"))]
    if not d:
        return None
    return {"mean_planted_minus_control": float(np.mean(d)), "plots_beyond_1_point": int(np.sum(np.abs(d) > 1.0)), "n_plots": len(d)}


def _exclusion_counts(S):
    by = Counter()
    for w in S.works:
        if not w.get("valid", True):
            by[("invalid", w["version"], w.get("writer"))] += 1
        for m in w.get("exclude_memes", []):
            by[(m, w["version"], w.get("writer"))] += 1
    return {"|".join(map(str, k)): v for k, v in sorted(by.items(), key=str)}


def _lowo(Ma, Mb, design, use, jcols, valid):
    out = {}
    for wr in sorted({d["writer"] for d in design if d["writer"]}):
        dd = [dict(d, version="_") if d["writer"] == wr else d for d in design]
        a = detection_aucs(Ma[:, [jcols[m] for m in use]], dd, use, REWORDED, valid)
        b = detection_aucs(Mb[:, [jcols[m] for m in use]], dd, use, REWORDED, valid)
        out[wr] = paired_meme_test(a, b, use, "greater")["stat"]
    return out


def _t2_secondaries(bym, scores, design, use, ids, jcols, valid, tier):
    sec = {}
    for model in MODELS:
        nk = f"neurons:{model}"
        if nk not in bym:
            continue
        j = f"judge:{model}"
        if j in bym:
            sec[f"S8_{j}_beats_{nk}"] = paired_meme_test(bym[j]["reworded"], bym[nk]["reworded"], use, "greater")
        sj = f"summary-judge:{model}"
        if sj in bym and j in bym:
            sec[f"S9_{sj}_below_{j}"] = paired_meme_test(bym[sj]["reworded"], bym[j]["reworded"], use, "less")
        for ctx_m in (f"hidden:{model}", f"judge:{model}"):
            for lex in (f"emb:{PRIMARY_ENCODER}", "keywords:expanded-qwen3-8b"):
                for a_ in (nk, ctx_m):
                    if a_ in bym and lex in bym:
                        sec[f"Scoded_{a_}_beats_{lex}"] = paired_meme_test(bym[a_]["coded"], bym[lex]["coded"], use, "greater")
        if f"{nk}:m1" in bym and f"{nk}:m5" in bym:
            t = paired_meme_test(bym[f"{nk}:m5"]["reworded"], bym[f"{nk}:m1"]["reworded"], use, "two-sided")
            t["single_unit_claim"] = bool(t["upper95_one_sided"] < 0.03)
            sec[f"m5_minus_m1_{model}"] = t
        if f"hidden:{model}" in bym:
            sec[f"key_{nk}_vs_hidden"] = paired_meme_test(bym[nk]["reworded"], bym[f"hidden:{model}"]["reworded"], use, "two-sided")
        for rv in bym:
            if rv != nk and not rv.startswith(nk):
                sec[f"{nk}|{rv}|reworded"] = paired_meme_test(bym[nk]["reworded"], bym[rv]["reworded"], use, "two-sided")
        embs = [k for k in bym if k.startswith(("emb:", "embwin:", "embdoc:")) and ":far" not in k]
        if embs:
            best = max(embs, key=lambda k: bym[k]["mean"]["reworded"] or -1)
            sec[f"best_encoder_hindsight_{model}"] = dict(paired_meme_test(bym[nk]["reworded"], bym[best]["reworded"], use, "two-sided"),
                                                         encoder=best, note="chosen with hindsight; favours the vectors")
        if f"length_resid:{model}" in bym:
            sec[f"length_residualised_{model}"] = {"mean_reworded": bym[f"length_resid:{model}"]["mean"]["reworded"]}
    for name, r in bym.items():
        sec[f"S10_{name}_implicit_below_banned"] = paired_meme_test(r["implicit"], r["banned"], use, "less")
        if name.startswith("keywords"):
            v = [r["decoy_vs_control"][m] for m in use]
            sec[f"S11_construction_{name}_decoy_above_control"] = signflip(v, alternative="greater", null_value=0.5)
    return {"secondary": sec}


def _t2_zh(memes, freeze):
    Z = _maybe("t2_stories_zh")
    if Z is None:
        return None
    ids = [m["id"] for m in memes]
    dz = _design(Z)
    vz = np.array([w.get("valid", True) for w in Z.works])
    fz = dict(freeze, t2=dict(freeze.get("t2", {}), keywords=freeze.get("t2", {}).get("keywords_zh", {})))
    out = {}
    for k, M in meme_story_scores(Z, memes, fz).items():
        a = detection_aucs(M, dz, ids, {"banned"}, vz, "twin")
        out[k] = {"banned": a, "mean": float(np.nanmean(list(a.values())))}
    nk = f"neurons:{PRIMARY_MODEL}"
    if nk in out:
        out["S12"] = signflip(list(out[nk]["banned"].values()), alternative="greater", null_value=0.5)
    return out


def _t2_natural(memes, freeze, min_pos=10, min_neg=10, min_kappa=0.4):
    """T2-N: work-level AUC per meme on exhaustively annotated natural plot summaries, with far-set detectors; memes
    need >= 10 positive and >= 10 negative works and annotator kappa >= 0.4 to enter the mean."""
    N_ = _maybe("t2n_works")
    if N_ is None or not all("memes_present" in w for w in N_.works):
        return None
    ids = [m["id"] for m in memes]
    lab = {w["id"]: set(w.get("memes_present", [])) for w in N_.works}
    kap = {}
    for m in ids:
        a = [m in set((w.get("annotators") or [[], []])[0]) for w in N_.works]
        b = [m in set((w.get("annotators") or [[], []])[1]) for w in N_.works]
        kap[m] = cohen_kappa(a, b) if any(a) or any(b) else None
    eligible = [m for m in ids if sum(m in lab[w] for w in N_.ids) >= min_pos and sum(m not in lab[w] for w in N_.ids) >= min_neg
                and (kap.get(m) is None or kap[m] >= min_kappa)]
    out = {"eligible_memes": eligible, "kappa": kap, "by_method": {}}
    for k, M in meme_story_scores(N_, memes, freeze, set_key="far").items():
        per = {}
        for j, mid in enumerate(ids):
            if mid in eligible:
                pos = [i for i, w in enumerate(N_.ids) if mid in lab[w]]
                neg = [i for i, w in enumerate(N_.ids) if mid not in lab[w]]
                per[mid] = auc(M[pos, j], M[neg, j])
        out["by_method"][k] = {"per_meme": per, "mean": float(np.nanmean(list(per.values()))) if per else None}
    nk = f"neurons:{PRIMARY_MODEL}:far"
    if nk in out["by_method"] and out["by_method"][nk]["per_meme"]:
        out["S13"] = signflip(list(out["by_method"][nk]["per_meme"].values()), alternative="greater", null_value=0.5)
    return out


def _t2_long(memes, freeze):
    """T2-long: 48 long texts of 12 stories; per-meme AUC on all texts and seconds per text for each method."""
    L = _maybe("t2_long")
    if L is None:
        return None
    ids = [m["id"] for m in memes]
    out = {"by_method": {}}
    for k, M in meme_story_scores(L, memes, freeze).items():
        per = {}
        for j, mid in enumerate(ids):
            pos = [i for i, w in enumerate(L.works) if mid in w["memes_present"]]
            neg = [i for i, w in enumerate(L.works) if mid not in w["memes_present"] and mid not in w.get("exclude_memes", [])]
            if len(pos) >= 3:
                per[mid] = auc(M[pos, j], M[neg, j])
        out["by_method"][k] = {"mean": float(np.nanmean(list(per.values()))) if per else None, "per_meme": per}
    for model in MODELS:
        a, b = out["by_method"].get(f"judge:{model}"), out["by_method"].get(f"summary-judge:{model}")
        if a and b and a["per_meme"]:
            out[f"summary_loss_{model}"] = signflip([b["per_meme"][m] - a["per_meme"][m] for m in a["per_meme"] if m in b["per_meme"]], alternative="less")
    return out


# =============================================================================================== T3: Batman-likeness across culture
def _cos_from_stats(d, q, cc, n, loo):
    """Cosine of each row (centred) with a centroid from its dots d = x.c, squared norms q = |x|^2 and |c|^2; rows in
    loo are compared with the centroid of the other n - 1 centroid rows."""
    out = d / (np.sqrt(np.maximum(q, 1e-24) * max(cc, 1e-24)))
    for i in loo:
        d2 = (n * d[i] - q[i]) / (n - 1)
        cc2 = (n * n * cc - 2 * n * d[i] + q[i]) / (n - 1) ** 2
        out[i] = d2 / np.sqrt(max(q[i], 1e-24) * max(cc2, 1e-24))
    return out


def _likeness_dense(V, groups, centre_rows, loo):
    V = np.asarray(V, np.float64)
    X = V - V[centre_rows].mean(0)
    q = (X ** 2).sum(1)
    out = {}
    for g, rows in groups.items():
        c = X[rows].mean(0)
        out[g] = _cos_from_stats(X @ c, q, float(c @ c), len(rows), loo.get(g, ()))
    return out


def _likeness_neurons(A_all, scale_rows, groups, centre_rows, loo, chunk=40000):
    """Centroid cosines on per-unit reference-standardised neuron scores, accumulated over unit chunks."""
    n, U = A_all.shape
    dots = {g: np.zeros(n) for g in groups}
    cn = {g: 0.0 for g in groups}
    q = np.zeros(n)
    for c0 in range(0, U, chunk):
        X = np.asarray(A_all[:, c0:c0 + chunk], np.float32)
        med, sd = unit_scale(X[scale_rows])
        X = (X - med) / sd
        X -= X[centre_rows].mean(0)
        X64 = X.astype(np.float64)
        q += (X64 ** 2).sum(1)
        for g, rows in groups.items():
            c = X64[rows].mean(0)
            dots[g] += X64 @ c
            cn[g] += float(c @ c)
    return {g: _cos_from_stats(dots[g], q, cn[g], len(groups[g]), loo.get(g, ())) for g in groups}


def _likeness_sparse(X, groups, centre_rows, loo):
    X = X.tocsr()
    mu = np.asarray(X[centre_rows].mean(0)).ravel()
    xm = np.asarray(X @ mu).ravel()
    q = np.asarray(X.multiply(X).sum(1)).ravel() - 2 * xm + mu @ mu
    out = {}
    for g, rows in groups.items():
        c = np.asarray(X[rows].mean(0)).ravel() - mu
        d = np.asarray(X @ c).ravel() - mu @ c
        out[g] = _cos_from_stats(d, q, float(c @ c), len(rows), loo.get(g, ()))
    return out


class _Stack:
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.shape = (a.shape[0] + b.shape[0], a.shape[1])

    def __getitem__(self, key):
        rows, cols = key
        assert rows == slice(None)
        return np.concatenate([np.asarray(self.a[:, cols]), np.asarray(self.b[:, cols])])


def run_t3(freeze, memes_file=PRE / "memes_for_tracing.json", pool_file=PRE / "culture_pool.json",
           relatives_file=PRE / "documented_relatives.json", n_perm=10000):
    """T3: whole-profile Batman-likeness (cosine to the centroid of the 32 DEV Batman works, centred on the reference
    films), superhero-likeness (cosine to the centroid of the pool's superhero-stratum negatives, leave-one-out for
    those negatives), Batman-specific likeness (Batman-likeness residualised on superhero-likeness over the negatives),
    and the orphan-vigilante layer (mean within-pool percentile of the Tier A far-set detectors). Entity-level
    inference against pseudo-franchises (entity_level_test)."""
    memes = json.load(open(memes_file))["memes"]
    tierA = [m for m in memes if m["tier"] == "A"]
    meta = json.load(open(pool_file))
    rel = {r["entity"]: r for r in json.load(open(relatives_file))["relatives"]}
    S, R = Set("t3_pool"), Set("t3_dev_masked")
    info = {w["title"]: w for w in meta["works"]}
    nS = len(S.ids)
    role = [info[t]["role"] for t in S.ids]
    entity = [info[t].get("entity") or t for t in S.ids]
    cluster = {i: info[t].get("neg_cluster") for i, t in enumerate(S.ids) if role[i] == "pool"}
    bat = [nS + i for i, w in enumerate(R.works) if w.get("group") == "batman"]
    centre = [nS + i for i, w in enumerate(R.works) if w.get("group") == "baseline"]
    rec = _recognised_ids(S, titles={t: t for t in S.ids}) or set()
    out = {"version": VERSION, "n_pool": nS, "roles": dict(Counter(role)), "by_method": {}, "strata_rule": {}}
    for skey in ("stratum", "stratum_category"):
        stratum = [info[t][skey] for t in S.ids]
        sup = [i for i in range(nS) if role[i] == "pool" and stratum[i] == "superhero"]
        groups = {"batman": bat, "superhero": sup}
        loo = {"superhero": sup}
        scores = {}
        for x in MODELS:
            try:
                A_ = _Stack(S.agg(x), R.agg(x))
            except (FileNotFoundError, AssertionError):
                continue
            scores[f"neurons:{x}"] = {k: v[:nS] for k, v in _likeness_neurons(A_, centre, groups, centre, loo).items()}
        X, vec = tfidf_matrix(S.texts() + R.texts())
        scores["tfidf"] = {k: v[:nS] for k, v in _likeness_sparse(X, groups, centre, loo).items()}
        L = lsa_vectors(X, X, k=100)
        scores["lsa"] = {k: v[:nS] for k, v in _likeness_dense(L, groups, centre, loo).items()}
        th, _ = lda_fit_transform(S.texts() + R.texts(), S.texts() + R.texts(), K=50)
        scores["lda"] = {k: v[:nS] for k, v in _likeness_dense(th, groups, centre, loo).items()}
        for enc in ENCODERS:
            if S.has(f"emb_{enc}.npy") and R.has(f"emb_{enc}.npy"):
                V = np.concatenate([work_embeddings(S, enc), work_embeddings(R, enc)])
                scores[f"emb:{enc}"] = {k: v[:nS] for k, v in _likeness_dense(V, groups, centre, loo).items()}
        neg = [i for i in range(nS) if role[i] == "pool"]
        for k in list(scores):
            b_, s_ = scores[k]["batman"], scores[k]["superhero"]
            Xf = np.column_stack([np.ones(len(neg)), s_[neg]])
            beta = np.linalg.lstsq(Xf, b_[neg], rcond=None)[0]
            scores[k]["batman_specific"] = b_ - (beta[0] + beta[1] * s_)
        for r in MODELS:
            rows = S.llm(r, "likeness")
            if rows:
                p = {x["work"]: _logit(x) for x in rows}
                scores[f"llm:{r}"] = {"batman": np.array([p.get(t, np.nan) for t in S.ids])}
        det = meme_story_scores(S, tierA, freeze, set_key="far")
        for k, M in det.items():
            if np.isnan(M).all() or k.startswith(("devunit", "length")):
                continue
            if any(k.startswith(p) for p in ("neurons:", "emb:", "hidden:", "keywords:logreg")) and not k.endswith(":far"):
                continue
            pct = np.column_stack([sps.rankdata(np.nan_to_num(M[:, j], nan=-np.inf)) / len(M) for j in range(M.shape[1])])
            scores.setdefault(f"layer|{k.replace(':far', '')}", {})["layer"] = pct.mean(1)
        negs = defaultdict(list)
        for i in neg:
            negs[stratum[i]].append(i)
        rtype = {i: (rel.get(entity[i]) or {}).get("relation_type") for i in range(nS) if role[i] == "relative"}
        typed = any(v for v in rtype.values())
        relp = [i for i in range(nS) if role[i] == "relative"]
        prim = [i for i in relp if rtype.get(i) in ("plot_role", "parody")] if typed else relp
        icon = [i for i in relp if rtype.get(i) in ("iconography_name", "meta", "transitive")]
        devf = lambda i: bool((rel.get(entity[i]) or {}).get("dev_franchise"))
        sets = {"S15_relatives_plot_fresh": [i for i in prim if not devf(i)], "S15b_relatives_plot_dev_franchise": [i for i in prim if devf(i)],
                "relatives_iconography_predicted_null": icon, "relatives_all": relp,
                "S16_book_tier1": [i for i, t in enumerate(S.ids) if info[t].get("book_tier") == 1],
                "S17_book_tier2": [i for i, t in enumerate(S.ids) if info[t].get("book_tier") == 2]}
        book_ent = [info[t].get("book_entity") or entity[i] for i, t in enumerate(S.ids)]
        lineage = [i for i in range(nS) if role[i] == "batman_lineage"]
        allpool = {s: neg for s in set(stratum)}
        blk = {}
        for k, sc in scores.items():
            r = {}
            if "batman" in sc:
                r["S14_positive_control_lineage_auc"] = stratified_auc(sc["batman"], lineage, allpool, stratum)[0]
            for layer in ("batman", "batman_specific", "layer"):
                if layer not in sc:
                    continue
                for sname, pos in sets.items():
                    if (layer == "layer") != sname.startswith(("S16", "S17")) and not (layer == "layer" and sname == "relatives_all"):
                        continue
                    ent = book_ent if sname.startswith(("S16", "S17")) else entity
                    t = entity_level_test(sc[layer], pos, ent, negs, stratum, cluster, n_perm=n_perm, rng=rng_for("t3ent", skey, k, layer, sname))
                    t["work_level_secondary"] = stratified_perm_test(sc[layer], pos, entity_weights(pos, ent) if pos else [], negs, stratum,
                                                                     n_perm=2000, rng=rng_for("t3w", k, layer, sname)) if pos else None
                    if skey == "stratum" and rec:
                        keep = [i for i in pos if S.ids[i] not in rec]
                        negs_u = {s_: [i for i in v if S.ids[i] not in rec] for s_, v in negs.items()}
                        t["unrecognised_only"] = _strip(entity_level_test(sc[layer], keep, ent, negs_u, stratum, cluster, n_perm=2000))
                    r[f"{sname}|{layer}"] = t
            if "batman_specific" in sc:
                contested = [i for i in range(nS) if role[i] == "contested"]
                r["contested_mean_percentile"] = float(np.nanmean([auc([sc["batman_specific"][i]], sc["batman_specific"][negs[stratum[i]]])
                                                                   for i in contested if negs.get(stratum[i])])) if contested else None
                mc = [i for i, t in enumerate(S.ids) if info[t].get("mike_candidate")]
                r["monte_cristo_mean_percentile"] = float(np.nanmean([auc([sc["batman_specific"][i]], sc["batman_specific"][negs[stratum[i]]])
                                                                      for i in mc if negs.get(stratum[i])])) if mc else None
            blk[k] = r
        con = {}
        for x in MODELS:
            nk = f"neurons:{x}"
            key = "S15_relatives_plot_fresh|batman_specific"
            if nk in blk and key in blk[nk]:
                for rv in ("tfidf", "lsa", f"emb:{PRIMARY_ENCODER}"):
                    if rv in blk and key in blk[rv]:
                        con[f"{nk}_vs_{rv}"] = entity_signflip(blk[nk][key]["per_entity"], blk[rv][key]["per_entity"], "greater")
        if skey == "stratum":
            out["by_method"] = blk
            out["directional_contrasts"] = con
            out["top50"] = {k: [S.ids[i] for i in np.argsort(-np.nan_to_num(sc.get("batman_specific", sc.get("batman", sc.get("layer"))), nan=-9))
                                if role[i] in ("pool", "relative")][:50] for k, sc in scores.items()}
            out["n_entities"] = {s: len({entity[i] for i in v}) for s, v in sets.items()}
            out["relation_types_coded"] = typed
        else:
            out["category_strata_sensitivity"] = {k: {kk: _strip(vv) for kk, vv in v.items() if kk.startswith("S15")} for k, v in blk.items()}
    out["bh"] = bh({f"{k}|{t}": v[t]["p"] for k, v in out["by_method"].items() for t in v if isinstance(v[t], dict) and "p" in v[t]})
    return out


# =============================================================================================== T4: mutation and selection
def run_t4(freeze, trio=T4_TRIO, n_perm=2000, n_sims=500):
    """T4 on the extended Batman lineage (all works re-run through the v2 name-removal protocol), neurons vs words."""
    S = Set("t4_batman")
    R = Set("dev_namefree")
    ref = [i for i, w in enumerate(R.works) if w.get("group") == "baseline"]
    yrs = [w["year"] for w in S.works]
    order = np.array(sorted(range(len(S.ids)), key=lambda i: (yrs[i], S.ids[i])))
    idx = S.index
    adapt = defaultdict(set)
    for t, srcs in BAT_ADAPTATIONS.items():
        for s in srcs:
            if t in idx and s in idx:
                adapt[idx[s]].add(idx[t])
    for t, (_, srcs) in FRESH_ADAPTATIONS.items():
        for s in srcs:
            if t in idx and s in idx:
                adapt[idx[s]].add(idx[t])
    trio_ids = [idx[t] for t in trio if t in idx]
    fresh_ids = {idx[t] for t in FRESH_ADAPTATIONS if t in idx}
    out = {"version": VERSION, "n_works": len(S.ids), "first_test_year": min((w["year"] for w in S.works if w.get("origin") == "test"), default=None)}
    reps = {}
    for x in MODELS:
        try:
            A_ = np.asarray(S.agg(x), np.float32)
            Ar = np.asarray(R.agg(x), np.float32)[ref]
        except (FileNotFoundError, AssertionError):
            continue
        reps[f"neurons:{x}"] = A_ > presence_bar(Ar, rank=1)
        reps[f"neurons:{x}|bar2"] = A_ > presence_bar(Ar, rank=2)
    Pw, _ = word_presence(S.texts(), S.texts() + [R.texts()[i] for i in ref])
    reps["words"] = Pw
    excess = {}
    for k, P in reps.items():
        r = {}
        for label, drop in (("all", set()), ("without_fresh_adaptations", fresh_ids)):
            ordr = np.array([i for i in order if i not in drop])
            r[label] = {"innovators": innovators_test(P, ordr, trio_ids, n_perm=n_perm, rng=rng_for("t4", k, label), adapt_of=adapt)}
        Po = np.asarray(P[order], bool)
        skip = _skip_for(order, adapt)
        ex = selection_excess(Po, skip, n_sims=n_sims, rng=rng_for("t4sel", k))
        excess[k] = ex
        r["selection"] = _strip(ex)
        n_mut, ms, later, fa = mutation_stats(Po, skip)
        sims = neutral_copying(Po.sum(1), n_mut, n_sims=min(n_sims, 200), rng=rng_for("neutral", k))
        mut = later[(fa >= 0) & (fa < len(order) - 1)]
        r["neutral_copying_secondary"] = {"gini_obs": gini(mut), "gini_null_p95": float(np.percentile(sims[:, 0], 95))}
        r["per_work"] = [{"work": S.ids[w], "year": yrs[w], "mutations": int(n_mut[i]),
                          "mean_success": None if not np.isfinite(ms[i]) else float(ms[i])} for i, w in enumerate(order)]
        out[k] = r
    for x in MODELS:
        nk = f"neurons:{x}"
        if nk in excess:
            out[f"S19_{nk}_vs_words"] = compare_excess(excess[nk], excess["words"])
            out[f"batch_effect_{nk}"] = batch_effect_auc(reps[nk], [w.get("origin", "dev") for w in S.works], yrs)
    return out


# =============================================================================================== T5: concept agreement
def run_t5(freeze):
    """T5: Llama partners chosen on DEV name-free text; held-out correlation on the sentences of the 1,000 random
    films of the T3 pool (primary) and on the T2 stories (secondary)."""
    t5 = freeze.get("t5")
    if not t5:
        return {"error": "no T5 partners in the freeze"}
    out = {"version": VERSION}
    for label, name, keep in (("random_films_primary", "t3_pool", lambda w: w.get("stratum") == "random" and w.get("role") == "pool"),
                              ("t2_stories_secondary", "t2_stories", lambda w: True)):
        S = _maybe(name)
        if S is None:
            continue
        try:
            q, qu = S.sent_units("qwen3-8b")
            l, lu = S.sent_units("llama3.1-8b")
        except FileNotFoundError:
            continue
        rows = np.where(np.isin(S.sent_work, [i for i, w in enumerate(S.works) if keep(w)]))[0]
        qc, lc = {int(u): i for i, u in enumerate(qu)}, {int(u): i for i, u in enumerate(lu)}

        def held(pairs):
            return np.array([corr_cols(np.asarray(q[rows][:, [qc[a]]], np.float32), np.asarray(l[rows][:, [lc[b]]], np.float32))[0]
                             for a, b in pairs])
        rc, rn = held(t5["concept_pairs"]), held(t5["null_pairs"])
        var = lambda u: float(np.var(np.asarray(q[rows][:, qc[u]], np.float32)))
        ratios = [var(a) / (np.mean([var(t5["null_pairs"][j][0]) for j in g]) + 1e-12) for (a, _), g in zip(t5["concept_pairs"], t5["null_groups"])]
        out[label] = {"test": t5_test(rc, rn, t5["null_groups"]), "r_concept": rc.tolist(), "r_null": rn.tolist(),
                      "median_concept": float(np.median(rc)), "median_null": float(np.median(rn)), "variance_ratio_concept_over_null": ratios,
                      "n_sentences": int(len(rows))}
    return out


# =============================================================================================== T6: interpretability
def t6_prepare(out_path=None, n_units=40, pool_set="t3_pool", dev_units_file=PRE / "dev_units.json"):
    """Builds the T6 blind label test (annotation_protocol.md A3) in two arms with one selection rule per arm for
    every kind: 'random' (units, topics or clusters drawn in seeded order) and 'best_of' (each kind's own quality
    order: DEV Batman enrichment for neurons, UMass coherence for LDA topics, mean membership probability for
    clusters); in both arms only units present in 1-20% of pool works qualify. Each unit gets its 16 top pool
    sentences (one per work), ranked by its own score among its members; ranks 1-8 go to the labeller, 9-16 to judges."""
    out_path = out_path or RES / "label_test_items.json"
    S = Set(pool_set)
    R = Set("t3_dev_masked")
    ref = [i for i, w in enumerate(R.works) if w.get("group") == "baseline"]
    du = json.load(open(dev_units_file))
    rng = rng_for("t6")
    kinds = {}

    def top_sentences(sent_scores, members=None):
        s = np.asarray(sent_scores, float).copy()
        if members is not None:
            s[~members] = -np.inf
        order = np.argsort(-s)
        seen, picks, texts = set(), [], set()
        for i in order:
            if not np.isfinite(s[i]):
                break
            w = S.sent_work[i]
            if w in seen or S.sentences[i] in texts:
                continue
            seen.add(w)
            texts.add(S.sentences[i])
            picks.append(int(i))
            if len(picks) == 16:
                break
        return picks
    band = lambda share: 0.01 <= share <= 0.20
    for model in MODELS:
        try:
            sv, units = S.sent_units(model)
            P = np.asarray(S.agg(model), np.float32)
            bar = presence_bar(np.asarray(R.agg(model), np.float32)[ref])
        except FileNotFoundError:
            continue
        col = {int(u): i for i, u in enumerate(units)}
        for arm, lst in (("random", du.get("label_test_random", {}).get(model, [])), ("best_of", du.get("label_test_best_of", {}).get(model, []))):
            chosen = [u for u in lst if u in col and band(float((P[:, u] > bar[u]).mean()))][:n_units]
            kinds[(f"neurons:{model}", arm)] = {str(u): top_sentences(np.asarray(sv[:, col[u]], np.float32)) for u in chosen}
    th, (cv, lda) = lda_fit_transform(S.texts() + R.texts(), S.texts() + R.texts(), K=100, seed=0)
    thS, thR = th[:len(S.ids)], th[len(S.ids):][ref]
    tbar = presence_bar(thR)
    prev = (thS > tbar[None]).mean(0)
    sent_th = lda.transform(cv.transform(S.sentences))
    coh = umass_coherence(lda.components_, cv.transform(S.texts()) > 0)
    ok_t = [t for t in range(100) if band(prev[t])]
    for arm, order in (("random", [int(x) for x in rng.permutation(100)]), ("best_of", [int(x) for x in np.argsort(-coh)])):
        kinds[("lda", arm)] = {str(t): top_sentences(sent_th[:, t]) for t in [t for t in order if t in ok_t][:n_units]}
    if S.has(f"emb_{PRIMARY_ENCODER}.npy"):
        E = np.asarray(S.emb(PRIMARY_ENCODER), np.float32)
        lab, cents, Rr, _, probs = cluster_topics(E)
        ks = sorted(set(lab) - {-1})
        prevc = {k: len(set(S.sent_work[lab == k])) / len(S.ids) for k in ks}
        okc = [k for k in ks if band(prevc[k]) and (lab == k).sum() >= 16]
        quality = {k: float(probs[lab == k].mean()) for k in okc}
        for arm, order in (("random", [ks[i] for i in rng.permutation(len(ks))] if ks else []), ("best_of", sorted(okc, key=lambda k: -quality[k]))):
            kinds[(f"topics:{PRIMARY_ENCODER}", arm)] = {str(k): top_sentences(probs, lab == k) for k in [k for k in order if k in okc][:n_units]}
    items = []
    for (kind, arm), units in kinds.items():
        keys = [u for u in units if len(units[u]) >= 10]
        for u in keys:
            held = units[u][8:16]
            for item in range(2):
                other = keys[(keys.index(u) + 1 + int(rng.integers(0, max(len(keys) - 1, 1)))) % len(keys)] if len(keys) > 1 else u
                own = held[int(rng.integers(0, len(held)))]
                oh = units[other][8:16]
                dec = oh[int(rng.integers(0, len(oh)))]
                a_first = bool(rng.random() < 0.5)
                items.append({"kind": kind, "arm": arm, "unit": u, "item": item, "A": S.sentences[own if a_first else dec],
                              "B": S.sentences[dec if a_first else own], "answer": "A" if a_first else "B"})
    doc = {"units": {f"{k}|{a}": {u: {"label_sentences": [S.sentences[i] for i in v[:8]]} for u, v in us.items()} for (k, a), us in kinds.items()},
           "items": items}
    write_json(doc, out_path)
    return doc


def word_overlap_floor(items_file=None, labels_file=None):
    """T6 floor: answer each judge item by the higher TF-IDF cosine between the label + description and the two
    sentences. Returns per-item correctness keyed (kind|arm, unit, item)."""
    items = json.load(open(items_file or RES / "label_test_items.json"))["items"]
    labels = json.load(open(labels_file or RES / "label_test_labels.json"))
    from sklearn.feature_extraction.text import TfidfVectorizer
    out = {}
    for it in items:
        lab = labels.get(f"{it['kind']}|{it.get('arm', 'random')}", labels.get(it["kind"], {})).get(it["unit"], {})
        text = f"{lab.get('label', '')} {lab.get('description', '')}"
        try:
            vec = TfidfVectorizer(stop_words="english").fit([text, it["A"], it["B"]])
            X = vec.transform([text, it["A"], it["B"]]).toarray()
            sa, sb = cosine(X[:1], X[1:2])[0, 0], cosine(X[:1], X[2:3])[0, 0]
        except ValueError:
            sa = sb = 0.0
        pick = "A" if sa > sb else "B" if sb > sa else "A"
        out[(f"{it['kind']}|{it.get('arm', 'random')}", it["unit"], it["item"])] = pick == it["answer"]
    return out


def run_t6(label_test_file=None, items_file=None, labels_file=None):
    """T6 interpretability: per kind and arm, accuracy with exact binomial CIs, share labelled 'unclear', judge kappa,
    McNemar against the word-overlap floor on the same items, Fisher tests of neurons against every other kind, and
    labeller calls per usable label."""
    label_test_file = Path(label_test_file or RES / "label_test_answers.json")
    if not label_test_file.exists():
        return {"error": "label test answers missing"}
    ans = json.load(open(label_test_file))
    labels = json.load(open(labels_file or RES / "label_test_labels.json")) if Path(labels_file or RES / "label_test_labels.json").exists() else {}
    try:
        floor = word_overlap_floor(items_file, labels_file)
    except FileNotFoundError:
        floor = {}
    by = defaultdict(list)
    per_item = defaultdict(dict)
    for a in ans:
        k = f"{a['kind']}|{a.get('arm', 'random')}"
        by[k].append(bool(a["correct"]))
        per_item[k].setdefault((a["unit"], a["item"]), []).append(bool(a["correct"]))
    out = {"version": VERSION, "kinds": {}}
    for k, v in by.items():
        n, x = len(v), sum(v)
        ci = sps.binomtest(x, n).proportion_ci(0.95) if n else None
        labs = labels.get(k, {})
        unclear = float(np.mean([str(l.get("label", "")).strip().lower() == "unclear" for l in labs.values()])) if labs else None
        pairs = [vv for vv in per_item[k].values() if len(vv) >= 2]
        kap = cohen_kappa([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else None
        majority = [(u, it, np.mean(vv) >= 0.5) for (u, it), vv in per_item[k].items()]
        fl = [(c, floor.get((k, u, it))) for u, it, c in majority if (k, u, it) in floor]
        mc = mcnemar_exact([a_ for a_, _ in fl], [b_ for _, b_ in fl]) if fl else None
        out["kinds"][k] = {"n": n, "correct": x, "acc": x / n if n else None, "ci95": [ci.low, ci.high] if ci else None,
                           "unclear_share": unclear, "judge_kappa": kap, "vs_floor_mcnemar": mc,
                           "floor_acc": float(np.mean([b_ for _, b_ in fl])) if fl else None,
                           "usable_labels": int(sum(str(l.get("label", "")).strip().lower() != "unclear" for l in labs.values())) if labs else None,
                           "labeller_calls_per_usable_label": (math.ceil(len(labs) / 10) / max(1, sum(str(l.get("label", "")).strip().lower() != "unclear"
                                                                                                   for l in labs.values()))) if labs else None}
    for nk in [k for k in by if k.startswith("neurons")]:
        for other in by:
            if other != nk and other.split("|")[1] == nk.split("|")[1]:
                a, b = by[nk], by[other]
                out[f"{nk}_vs_{other}"] = {"fisher_p": float(sps.fisher_exact([[sum(a), len(a) - sum(a)], [sum(b), len(b) - sum(b)]])[1])}
    return out


# =============================================================================================== cost
COST_METHODS = ["neurons", "neurons+hidden", "emb:qwen3emb-4b", "emb:e5-large", "emb:e5-base", "judge", "judge-shared", "judge-1pass",
                f"judge-small:{SMALL_READER}", "summary", "summary-judge", "nli", "keywords", "tfidf", "lsa", "lda"]


def total_cost(rec, N, M, Q=0):
    """Total seconds to score N stories on M memes and then answer Q new memes later (re-scoring the N stories):
    setup (fixed + per meme) + passes + per-meme work. Neurons store every unit's aggregate, so a later meme costs
    one unit selection and re-scoring from storage; an LLM judge re-reads every story for a later meme (or answers it
    from a cached prefix if the prefix was kept); embeddings and keywords re-score from stored vectors or text."""
    s = rec.get("setup_fixed_s", 0.0) + M * rec.get("setup_per_meme_s", 0.0) + N * rec.get("per_story_s", 0.0)
    s += N * M * rec.get("per_story_per_meme_s", 0.0)
    s += Q * (rec.get("setup_per_meme_s", 0.0) + N * rec.get("new_meme_per_story_s", rec.get("per_story_per_meme_s", 0.0)))
    return float(s)


def dominates(a, b):
    """Frontier rule (PREREGISTRATION.md T6): A dominates B when A's accuracy lower 95% bound is at least B's point
    estimate and A costs no more per story."""
    return bool(a["auc_lower95"] >= b["auc"] and a["cost_per_story_s"] <= b["cost_per_story_s"])


def run_cost(t2=None, bench_file=None, sets=None):
    """T6 cost accounting: per-method cost recipes from the controlled benchmark (bench.py, DEV texts) and the
    production cost files, total-cost curves C(N, M, Q), storage under three retention policies, and the
    accuracy-cost frontier on reworded planted stories with the dominance rule (H6)."""
    bench_file = Path(bench_file or RES / "bench.json")
    bench = json.load(open(bench_file)) if bench_file.exists() else {}
    out = {"version": VERSION, "production": {}, "recipes": bench.get("recipes", {}), "bench_meta": bench.get("meta")}
    for d in sorted(FEAT.glob("*")) if sets is None else [FEAT / s for s in sets]:
        if not d.is_dir():
            continue
        rows = {f.stem[5:]: json.load(open(f)) for f in d.glob("cost_*.json")}
        n_works = len(read_jsonl(DATA / f"{d.name}.jsonl")) if (DATA / f"{d.name}.jsonl").exists() else None
        for v in rows.values():
            if n_works:
                v["seconds_per_work"] = v.get("seconds", 0) / n_works
        out["production"][d.name] = rows
    t2 = t2 if t2 is not None else (json.load(open(RES / "t2.json")) if (RES / "t2.json").exists() else {})
    rec = out["recipes"]
    n_m = len(json.load(open(PRE / "memes_for_tracing.json"))["memes"])
    N = t2.get("n_stories", 816)
    out["curves"] = {m: {f"N{n}_M{mm}_Q{q}": total_cost(r, n, mm, q) for n in (100, 1000, 100000) for mm in (1, 10, n_m) for q in (0, 10)}
                     for m, r in rec.items()}
    out["storage_policies"] = {m: {"all_units_bytes_per_story": r.get("store_all_bytes"), "selected_only_bytes_per_story": r.get("store_selected_bytes"),
                                   "none": 0} for m, r in rec.items()}
    front = []
    for name, r in t2.get("by_method", {}).items():
        fam = _cost_family(name)
        if fam not in rec:
            continue
        per = [v for v in r["reworded"].values() if np.isfinite(v)]
        if not per:
            continue
        ci = mean_ci(per, key=f"front{name}")
        front.append({"method": name, "family": fam, "auc": float(np.mean(per)), "auc_lower95": ci["lower95_one_sided"],
                      "cost_per_story_s": total_cost(rec[fam], N, n_m) / N})
    out["frontier"] = front
    out["dominated_by"] = {a["method"]: [b["method"] for b in front if b is not a and dominates(b, a)] for a in front}
    out["H6"] = {x: {"neurons_method": f"neurons:{x}", "dominated_by": out["dominated_by"].get(f"neurons:{x}")} for x in MODELS}
    return out


def _cost_family(name):
    if name.startswith(("neurons:", "devunit:", "length_resid:")):
        return "neurons"
    if name.startswith("hidden:"):
        return "neurons+hidden"
    if name.startswith(("emb:", "embwin:", "embdoc:")):
        return "emb:" + name.split(":")[1]
    if name.startswith("judge-shared"):
        return "judge-shared"
    if name.startswith("judge-1pass"):
        return "judge-1pass"
    if name.startswith(f"judge:{SMALL_READER}"):
        return f"judge-small:{SMALL_READER}"
    if name.startswith("judge:") and "claude" not in name:
        return "judge"
    if name.startswith("summary-judge"):
        return "summary-judge"
    if name.startswith("nli"):
        return "nli"
    if name.startswith("keywords"):
        return "keywords"
    if name in ("lda",) or name.startswith("topics"):
        return "lda"
    return name


# =============================================================================================== DEV freeze (F1)
def far_screen(memes, ex_set, ref_sentences_emb, bat_emb, enc_name):
    """F1 screen of the far example sets (annotation_protocol.md A8): an example is dropped when its maximum cosine to
    DEV Batman and cousin sentences exceeds its maximum cosine to reference-film sentences; at least 3 are kept (the
    3 with the smallest excess)."""
    E = np.asarray(ex_set.emb(enc_name), np.float32)
    keep = {}
    for m in memes:
        rows = [i for i, w in enumerate(ex_set.works) if w["meme"] == m["id"] and w.get("set") == "far"]
        if not rows:
            continue
        ex_ = E[rows]
        excess = (ex_ @ bat_emb.T).max(1) - (ex_ @ ref_sentences_emb.T).max(1)
        ok = [rows[k] for k in range(len(rows)) if excess[k] <= 0]
        if len(ok) < 3:
            ok = [rows[k] for k in np.argsort(excess)[:3]]
        keep[m["id"]] = sorted(ok)
    return keep


def far_screen_tfidf(examples, bat_sentences, ref_sentences, keep_min=3):
    """P1 screen of a meme's far examples (annotation_protocol.md A8), TF-IDF version: an example is dropped when its
    maximum TF-IDF cosine to DEV Batman and cousin sentences exceeds its maximum to reference-film sentences; at least
    keep_min are kept (those with the smallest excess). Returns (kept indices, excess per example)."""
    X, _ = tfidf_matrix([lexical_text([s]) for s in list(examples) + list(bat_sentences) + list(ref_sentences)], min_df=1)
    k, nb = len(examples), len(bat_sentences)
    E, Bt, Rf = X[:k], X[k:k + nb], X[k + nb:]
    ex = np.asarray((E @ Bt.T).max(1).todense()).ravel() - np.asarray((E @ Rf.T).max(1).todense()).ravel()
    keep = [i for i in range(k) if ex[i] <= 0]
    if len(keep) < keep_min:
        keep = sorted(np.argsort(ex)[:keep_min].tolist())
    return keep, ex.tolist()


FAR_FORBIDDEN = re.compile(r"\b(gotham|crime-ridden city|california|spanish colonial|revolution|revolutionary|guillotine|billionaire|"
                           r"inventor|costume|costumed|caped|cape|vigilante|masked avenger|bat)\b", re.I)


def dev_freeze(out_path=PRE / "dev_freeze.json", memes_file=PRE / "memes_for_tracing.json", dev_units_file=PRE / "dev_units.json",
               unit_list_dir=PRE, n_perm=N_PERM_LINEAGE):
    """Freeze F1. Everything that defines a method is fixed here on DEV inputs and the frozen examples only, before
    any TEST text is processed, with the same selection budget for every method family:
      - T1a: every configuration of every lineage family (LINEAGE_GRID, at most 4 each) on the DEV name-free E2; each
        family's configuration with the lowest statistic is frozen (ties: the first listed), the defaults are kept as
        a sensitivity analysis; the DEV reproduction check; the codes-only baseline
      - T2 detectors for three example sets (main = definition + explicit + reworded; explicit; far = definition +
        far), each against the contrastive background (reference-film sentences, the other memes' sentences and the
        meme's own hard negatives), two variants per family chosen by leave-one-example-out on the main set:
        neurons m1 / m5; embeddings and hidden states contrastive prototype / L2 probe; LDA K = 50 / 100; plus the
        lexical probe, the keyword queries (TF-IDF, TF-IDF + WordNet, the LLM analysts with and without the
        evasion-aware prompt, the ban-list-aligned Claude query) with their ban-list overlap, and the m1ctx variant
      - the far-set screen, the in-context check of the selected units, per-unit length slopes
      - T5 Llama partners; the per-sentence unit list; timers for every step."""
    timers = {}
    memes = json.load(open(memes_file))["memes"]
    ids = [m["id"] for m in memes]
    du = json.load(open(dev_units_file))
    ex, bg = Set("t2_examples"), Set("t2_background")
    exw = ex.works
    nbg = len(bg.sentences)
    fz = {"version": VERSION, "created": time.strftime("%Y-%m-%d %H:%M"), "t2": {"neurons": {}, "neurons_selected": {}, "neurons_ctx": {},
          "devunits": {}, "keywords": {}, "emb": {}, "emb_selected": {}, "hidden": {}, "length_slopes": {}, "loeo": {}},
          "timers": timers}
    t2 = fz["t2"]
    t2["dev_class"] = {m: (du["t2_mapped"].get(m) or {}).get("dev_class") for m in ids}
    t2["dev_survivor"] = {m: (du["t2_mapped"].get(m) or {}).get("dev_rewording_survivor") for m in ids}
    # far-set screen (e5-base, DEV original Batman + cousin sentences vs reference-film sentences)
    far_keep = None
    dev_o = _maybe("dev_original")
    if dev_o is not None and ex.has("emb_e5-base.npy") and dev_o.has("emb_e5-base.npy"):
        with timed(timers, "far_screen"):
            Eo = np.asarray(dev_o.emb("e5-base"), np.float32)
            gr = np.array([dev_o.works[i].get("group") for i in dev_o.sent_work])
            far_keep = far_screen(memes, ex, Eo[gr == "baseline"], Eo[gr != "baseline"], "e5-base")
    t2["far_kept_rows"] = far_keep
    rows_of = defaultdict(list)
    for i, w in enumerate(exw):
        if w.get("set") == "far" and far_keep is not None and i not in set(far_keep.get(w["meme"], [])):
            continue
        rows_of[(w["meme"], w.get("set"))].append(i)
    pos = {(m, st): [r for s_ in sets for r in rows_of[(m, s_)]] for m in ids for st, sets in EXAMPLE_SETS.items()}
    own = {m: [i for i, w in enumerate(exw) if w["meme"] == m and w.get("set") in POSITIVE_ROWS] for m in ids}
    bg_rows_for = {m: np.r_[np.arange(nbg), nbg + np.array([i for i in range(len(exw)) if i not in set(own[m])], int)] for m in ids}
    # ---------------- neurons
    for model in MODELS:
        if not (ex.has(f"neurons_{model}/full.npy") and bg.has(f"neurons_{model}/full.npy")):
            continue
        with timed(timers, f"neurons_select_{model}"):
            B = RowCat([bg.full(model), ex.full(model)])
            prow = {k: [nbg + i for i in v] for k, v in pos.items() if v}
            orow = {m: [nbg + i for i in v] for m, v in own.items()}
            det = neuron_detectors(B, prow, orow)
            crit = neuron_loeo_criteria(B, {k: v for k, v in prow.items() if k[1] == "main"}, orow,
                                        {k: v for k, v in det.items() if k[1] == "main"})
        store = {}
        for (m, st), d in det.items():
            for variant, us in (("m1", [d["m1"]]), ("m5", d["m5"])):
                q50, sc = background_standardiser(B, us, drop=orow[m])
                store.setdefault(st, {}).setdefault(m, {})[variant] = {"units": [int(u) for u in us], "q50": q50.tolist(), "scale": sc.tolist()}
        t2["neurons"][model] = store
        cm1 = float(np.mean([v["m1"] for v in crit.values()]))
        cm5 = float(np.mean([v["m5"] for v in crit.values()]))
        t2["loeo"][f"neurons:{model}"] = {"m1": cm1, "m5": cm5}
        t2["neurons_selected"][model] = "m5" if cm5 > cm1 else "m1"
        sel_units = sorted({u for st in store for m in store[st] for v in store[st][m].values() for u in v["units"]})
        X = np.asarray(bg.full(model)[:, sel_units], np.float32)
        lw = np.log1p([len(s.split()) for s in bg.sentences])
        lw_c = lw - lw.mean()
        t2["length_slopes"][model] = {str(u): float(((X[:, k] - X[:, k].mean()) @ lw_c) / (lw_c @ lw_c + 1e-12)) for k, u in enumerate(sel_units)}
        if dev_o is not None and dev_o.has(f"neurons_{model}/full.npy"):
            with timed(timers, f"in_context_check_{model}"):
                F = dev_o.full(model)
                ref_rows = np.where(np.isin(dev_o.sent_work, [i for i, w in enumerate(dev_o.works) if w.get("group") == "baseline"]))[0]
                main_units = sorted({u for m in ids for u in store.get("main", {}).get(m, {}).get(t2["neurons_selected"][model], {}).get("units", [])})
                iso = np.asarray(bg.full(model)[:, main_units], np.float32)
                ctxv = np.asarray(F[ref_rows][:, main_units], np.float32)
                rho = [float(sps.spearmanr(iso[:, k], ctxv[:, k])[0]) if iso.shape[0] == ctxv.shape[0] else None for k in range(len(main_units))]
                t2.setdefault("in_context_check", {})[model] = {"median_spearman": float(np.nanmedian([r for r in rho if r is not None])) if rho else None,
                                                                "per_unit": dict(zip(map(str, main_units), rho))}
            ctx = _maybe("t2_examples_ctx")
            if ctx is not None and ctx.has(f"neurons_{model}/full.npy"):
                with timed(timers, f"m1ctx_{model}"):
                    Bc = RowCat([np.asarray(F[ref_rows], np.float32), ctx.full(model)])
                    nref = len(ref_rows)
                    cw = ctx.works
                    pc = {(m, "main"): [nref + i for i, w in enumerate(cw) if w["meme"] == m and w.get("set") in EXAMPLE_SETS["main"]] for m in ids}
                    oc = {m: [nref + i for i, w in enumerate(cw) if w["meme"] == m and w.get("set") in POSITIVE_ROWS] for m in ids}
                    dc = neuron_detectors(Bc, {k: v for k, v in pc.items() if v}, oc, folds=False)
                    for (m, _), d in dc.items():
                        q50, sc = background_standardiser(Bc, [d["m1"]], drop=oc[m])
                        t2["neurons_ctx"].setdefault(model, {})[m] = {"units": [int(d["m1"])], "q50": q50.tolist(), "scale": sc.tolist()}
        t2["devunits"][model] = {m: (du["t2_mapped"].get(m) or {}).get(model) for m in ids}
    # ---------------- lexical: keyword queries and the lexical probe
    with timed(timers, "keywords_and_logreg"):
        bgs = list(bg.sentences)
        exs = [w["sentences"][0] for w in exw]
        cbg = {m: bgs + [exs[i] for i in range(len(exw)) if i not in set(own[m])] for m in ids}
        t2["keywords"]["tfidf"] = {m: tfidf_keywords([exs[i] for i in pos[(m, "main")]], cbg[m]) for m in ids}
        wn_used = True
        t2["keywords"]["tfidf-wordnet"] = {}
        for m in ids:
            t2["keywords"]["tfidf-wordnet"][m], ok = wordnet_expand(t2["keywords"]["tfidf"][m])
            wn_used &= ok
        for src in ("qwen3-8b", "llama3.1-8b", "expanded-qwen3-8b", "expanded-llama3.1-8b", "ban-list-aligned"):
            p = FEAT / "t2_examples" / f"keywords_{src}.json"
            if p.exists():
                q = json.load(open(p))
                if src.startswith("expanded"):
                    q = {m: wordnet_expand(v)[0] for m, v in q.items()}
                t2["keywords"][src] = q
        t2["wordnet_available"] = wn_used
        for src in ("qwen3-8b", "expanded-qwen3-8b", "tfidf"):
            p = FEAT / "t2_examples" / f"keywords_{src}_zh.json"
            if p.exists():
                t2.setdefault("keywords_zh", {})[src] = json.load(open(p))
        ban = {m["id"]: m["keywords"] for m in memes}
        t2["ban_list_overlap"] = {src: {m: ban_list_overlap(q.get(m, []), ban[m]) for m in ids} for src, q in t2["keywords"].items()}
        vec = fit_lexical_vectorizer(bgs + exs)
        t2["logreg"] = {"vocab": {k: int(v) for k, v in vec.vocabulary_.items()}, "idf": vec.idf_.tolist(), "det": {}}
        for st in ("main", "far"):
            t2["logreg"]["det"][st] = {m: fit_keyword_logreg([exs[i] for i in pos[(m, st)]], cbg[m], vec) for m in ids if pos[(m, st)]}
    # ---------------- embeddings and hidden states: contrastive prototype vs probe, chosen by leave-one-example-out
    for enc in ENCODERS:
        if not (ex.has(f"emb_{enc}.npy") and bg.has(f"emb_{enc}.npy")):
            continue
        with timed(timers, f"emb_detectors_{enc}"):
            E = np.concatenate([np.asarray(bg.emb(enc), np.float32), np.asarray(ex.emb(enc), np.float32)])
            by, crit = {}, {"cproto": [], "probe": []}
            for st in EXAMPLE_SETS:
                for m in ids:
                    if not pos[(m, st)]:
                        continue
                    P = E[[nbg + i for i in pos[(m, st)]]]
                    Bm = E[bg_rows_for[m]]
                    by.setdefault(st, {})[m] = {k: fit_linear(k, P, Bm) for k in ("cproto", "probe")}
                    if st == "main":
                        for k in crit:
                            crit[k].append(loeo_linear(k, P, Bm))
            t2["emb"][enc] = by
            t2["loeo"][f"emb:{enc}"] = {k: float(np.mean(v)) for k, v in crit.items()}
            t2["emb_selected"][enc] = max(("cproto", "probe"), key=lambda k: (t2["loeo"][f"emb:{enc}"][k], k == "cproto"))
    # lineage grid on DEV (T1a); the hidden-state layer is part of it
    dev = Set("dev_namefree")
    with timed(timers, "t1a_grid"):
        grid = [m for m in lineage_methods(llm=True, hidden=True, grid=True)]
        spec = assemble_batman_spec("t1a_dev", dev, dev, None, dev, grid)
        res = run_lineage(spec, grid, n_perm=n_perm, residual=True)
    fz["t1a_dev"] = {m: {k: v for k, v in r.items() if k in ("stat", "p", "null_mean", "per_target_share", "error", "r2_of_fit")} for m, r in res.items()}
    sel = {}
    for m, r in res.items():
        if "|" in m or "stat" not in r:
            continue
        name, kind, arg, cfg = _family(m)
        order = LINEAGE_GRID[kind]
        if name not in sel or (r["stat"], order.index(cfg)) < (res[f"{name}@{sel[name]}"]["stat"], order.index(sel[name])):
            sel[name] = cfg
    fz["lineage_selected"] = sel
    fz["dev_reproduction"] = {}
    for model, expected in (("qwen3-8b", 0.164), ("llama3.1-8b", 0.102)):
        m = f"neurons:{model}@base"
        if m in spec["Sim"] and not isinstance(spec["Sim"][m], dict):
            r = lineage_test(spec["Sim"][m], spec["targets"], spec["sources"], spec["eligible"], spec["hub"], n_perm=2000, ties="strict")
            fz["dev_reproduction"][model] = {"stat": r["stat"], "expected": expected, "ok": bool(abs(r["stat"] - expected) <= 0.03)}
    for model in MODELS:
        if not (bg.has(f"neurons_{model}/hidden.npy") and ex.has(f"neurons_{model}/hidden.npy")):
            continue
        li = int(sel.get(f"hidden:{model}", "L1")[1:])
        with timed(timers, f"hidden_detectors_{model}"):
            Hb = np.asarray(bg.hidden(model)[:, li], np.float32)
            mu, sd = Hb.mean(0), Hb.std(0) + 1e-6
            H = (np.concatenate([Hb, np.asarray(ex.hidden(model)[:, li], np.float32)]) - mu) / sd
            by, crit = {}, {"cproto": [], "probe": []}
            for st in EXAMPLE_SETS:
                for m in ids:
                    if not pos[(m, st)]:
                        continue
                    P = H[[nbg + i for i in pos[(m, st)]]]
                    Bm = H[bg_rows_for[m]]
                    by.setdefault(st, {})[m] = {k: fit_linear(k, P, Bm) for k in ("cproto", "probe")}
                    if st == "main":
                        for k in crit:
                            crit[k].append(loeo_linear(k, P, Bm))
            t2["loeo"][f"hidden:{model}"] = {k: float(np.mean(v)) for k, v in crit.items()}
            t2["hidden"][model] = {"layer": li, "mu": mu.tolist(), "sd": sd.tolist(), "det": by,
                                   "selected": max(("cproto", "probe"), key=lambda k: (t2["loeo"][f"hidden:{model}"][k], k == "cproto"))}
    # LDA K by leave-one-example-out
    with timed(timers, "lda_K"):
        texts = bgs + exs
        critK = {}
        for K in (50, 100):
            th, _ = lda_fit_transform(texts, texts, K=K)
            vals = []
            for m in ids:
                P = [nbg + i for i in pos[(m, "main")]]
                Bm = bg_rows_for[m]
                for k_ in range(len(P)):
                    rest = [p for j, p in enumerate(P) if j != k_]
                    t = int(np.argmax(th[rest].mean(0) - th[Bm].mean(0)))
                    vals.append(float(_pct_of(th[[P[k_]], t], th[Bm, t])[0]))
            critK[K] = float(np.mean(vals)) if vals else 0.0
        t2["loeo"]["lda"] = critK
        t2["lda_K"] = max(critK, key=lambda k: (critK[k], -k))
    t2["nli_memes"] = ids
    # T5 partners (DEV name-free sentences)
    if dev.has("neurons_llama3.1-8b/full.npy") and dev.has("neurons_qwen3-8b/full.npy"):
        with timed(timers, "t5_partners"):
            Qf, Lf = dev.full("qwen3-8b"), dev.full("llama3.1-8b")
            concepts = du["t5_concepts"]
            c_units = [c["qwen_unit"] for c in concepts]
            n_units = [u for c in concepts for u in c["null_units"]]
            bu_c, br_c = best_partner(np.asarray(Qf[:, c_units], np.float32), Lf)
            bu_n, br_n = best_partner(np.asarray(Qf[:, n_units], np.float32), Lf)
            k = len(concepts[0]["null_units"])
            fz["t5"] = {"concept_pairs": [[a, int(b)] for a, b in zip(c_units, bu_c)], "concept_dev_r": br_c.tolist(),
                        "null_pairs": [[a, int(b)] for a, b in zip(n_units, bu_n)], "null_dev_r": br_n.tolist(),
                        "null_groups": [list(range(i * k, (i + 1) * k)) for i in range(len(c_units))]}
    ul = {m: set() for m in MODELS}
    for model, store in t2["neurons"].items():
        for st in store.values():
            for d in st.values():
                for v in d.values():
                    ul[model].update(v["units"])
    for model, d in t2["neurons_ctx"].items():
        for v in d.values():
            ul[model].update(v["units"])
    for model, d in t2["devunits"].items():
        ul[model].update(int(u) for u in d.values() if u is not None)
    if "t5" in fz:
        ul["qwen3-8b"].update(a for a, _ in fz["t5"]["concept_pairs"] + fz["t5"]["null_pairs"])
        ul["llama3.1-8b"].update(b for _, b in fz["t5"]["concept_pairs"] + fz["t5"]["null_pairs"])
    for model in MODELS:
        ul[model].update(du.get("label_test_best_of", {}).get(model, []))
        ul[model].update(du.get("label_test_random", {}).get(model, []))
    fz["sentence_units"] = {m: sorted(int(u) for u in v) for m, v in ul.items()}
    write_json(fz, out_path)
    for model, units in fz["sentence_units"].items():
        np.save(Path(unit_list_dir) / f"sentence_units_{model}.npy", np.array(units, np.int64))
    return fz


# =============================================================================================== primary (H1-H6)
INTERPRETATION = {
    "H1": "Supported: neuron memes keep tracking a meme when its usual words are removed better than an analyst who expects "
          "evasion and better than a lexical classifier. Falsified: keywords chosen with evasion in mind (or a lexical probe) "
          "do as well, so the fuzziness claim does not separate neurons from good keyword practice.",
    "H2": "Supported: neurons are within 0.05 AUC of the strongest sentence, window and same-family embeddings (superiority "
          "only if its step passes). Falsified: embeddings are better by more than 0.05; the case for neurons then rests on "
          "interpretability (T6) and cost (H6), not accuracy.",
    "H3": "Supported: units found on DEV Batman text respond specifically to their own meme in unrelated genres. Falsified: "
          "they respond to 'something dramatic happens' as much as to their meme.",
    "H4": "Supported: under heavy vocabulary drift, neurons recover documented sources better than TF-IDF and LSA and still "
          "recover them above chance. Falsified: they do no better than word methods once the words change.",
    "H5": "Supported: on renamed Shakespeare adaptations neurons recover the source play better than TF-IDF and LSA. "
          "Falsified: word methods do as well. Uninformative when no method recovers the sources.",
    "H6": "Not dominated: no method is both at least as accurate (lower bound) and no more expensive per story on reworded "
          "planted memes. Dominated: a cheaper method is as accurate; the neuron case then rests on interpretability."}


def run_primary(t1=None, t2=None, freeze=None, cost=None):
    """H1-H5 with the section-7 checks (an inconclusive test enters Holm with p = 1), Holm across the five in Qwen3-8B,
    three-way verdicts against the smallest effects of interest, the H2 superiority step, H6 (frontier decision),
    and the Llama replication."""
    t1 = t1 or json.load(open(RES / "t1.json"))
    t2 = t2 or json.load(open(RES / "t2.json"))
    if freeze is None and (PRE / "dev_freeze.json").exists():
        freeze = json.load(open(PRE / "dev_freeze.json"))
    freeze = freeze or {}
    if cost is None and (RES / "cost.json").exists():
        cost = json.load(open(RES / "cost.json"))
    bym = t2.get("by_method", {})
    checks = {}
    rep = freeze.get("dev_reproduction", {})
    checks["dev_reproduction_ok"] = {m: v.get("ok") for m, v in rep.items()}
    audit = t2.get("human_audit")
    checks["t2_human_audit"] = audit
    kw = (bym.get("keywords:qwen3-8b") or {}).get("mean", {}).get("explicit")
    checks["t2_keywords_explicit_twin_auc"] = kw
    xf = t2.get("cross_family_check")
    checks["t2_cross_family_check"] = xf
    if audit is not None:
        audit_fail = not audit.get("pass")
    else:
        audit_fail = xf is not None and not xf.get("pass")
        checks["t2_note"] = "no human audit: the cross-family reader's pass rule is used"
    t2_bad = bool(audit_fail or (kw is not None and kw < 0.75))
    man = t1.get("t1b_manipulation", {})
    rm100 = (man.get(100) or man.get("100") or {}).get("banned_token_share_removed")
    tp = t1.get("t1b_tfidf_pooled_share", {})
    tf0, tf100 = tp.get(0, tp.get("0")), tp.get(100, tp.get("100"))
    checks["t1b_removal_at_100"] = rm100
    checks["t1b_tfidf_pooled_share_0_100"] = [tf0, tf100]
    checks["t1b_levels_merged"] = man.get("levels_merged")
    h4_bad = bool((rm100 is not None and rm100 < 0.90) or (tf0 is not None and tf100 is not None and not tf100 > tf0))
    pc = (t1.get("t1d_positive_control", {}).get(resolve("tfidf")) or {}).get("stat")
    checks["t1d_positive_control_share"] = pc
    h5_bad = pc is not None and pc > 0.10
    informative = bool(t1.get("t1d_informative_methods"))
    checks["t1d_informative"] = informative
    out = {"version": VERSION, "checks": checks, "models": {}, "interpretation": INTERPRETATION}
    for model in MODELS:
        repro_bad = rep.get(model, {}).get("ok") is False
        H = {"H1": t2.get("H1", {}).get(model, {}), "H2": t2.get("H2", {}).get(model, {}), "H3": t2.get("H3", {}).get(model, {}),
             "H4": t1.get("H4", {}).get(model, {}), "H5": t1.get("H5", {}).get(model, {})}
        h4 = H["H4"]
        h4_p = h4.get("p_iut", np.nan)
        if h4.get("source_leak_confounded") and "without_leaking" in h4:
            h4_p = h4["without_leaking"].get("p_iut", np.nan)
        p = {"H1": H["H1"].get("p_iut", np.nan), "H2": H["H2"].get("p_iut", np.nan), "H3": H["H3"].get("p", np.nan), "H4": h4_p,
             "H5": H["H5"].get("p_iut", np.nan)}
        bad = {"H1": t2_bad, "H2": t2_bad, "H3": t2_bad, "H4": h4_bad, "H5": h5_bad or not informative}
        bad = {k: bool(v or repro_bad or not np.isfinite(p[k] if p[k] is not None else np.nan)) for k, v in bad.items()}
        p_used = {k: (1.0 if bad[k] else float(p[k])) for k in p}
        entry = {"p": p, "p_entered": p_used, "inconclusive": bad}
        if model == PRIMARY_MODEL:
            hm = holm(p_used)
            entry["holm"] = hm
            sig = {k: hm[k]["reject"] for k in p}
            thr = {k: hm[k]["threshold"] for k in p}
        else:
            sig = {k: bool(p_used[k] < ALPHA) for k in p}
            thr = {k: ALPHA for k in p}
            entry["replicates"] = sig
        extra = {"H1": H["H1"].get("neutral_consistent", True), "H2": H["H2"].get("neutral_consistent", True), "H3": True,
                 "H4": H["H4"].get("conjunct_ok", False), "H5": True}
        verdict = {}
        for k in p:
            note = None
            if k == "H5" and not informative:
                verdict[k] = "uninformative"
                continue
            supported = sig[k] and extra[k]
            if sig[k] and not extra[k]:
                note = "test passed but its required condition failed"
            if k in ("H1", "H2") and supported and H[k].get("itt_p_iut", 0) > thr[k]:
                note = "per-protocol only"
            if k == "H4" and h4.get("source_leak_confounded"):
                note = "source-leak sensitivity used"
            verdict[k] = verdict3(supported, bool(H[k].get("falsified")), bad[k], note)
        entry["verdict"] = verdict
        sp = H["H2"].get("superiority_p_iut")
        entry["H2_superiority_supported"] = bool(verdict["H2"].startswith("supported") and sp is not None and sp <= thr["H2"])
        dom = ((cost or {}).get("H6") or {}).get(model, {}).get("dominated_by")
        entry["H6"] = {"dominated_by": dom, "verdict": None if dom is None else ("dominated" if dom else "not dominated")}
        entry["estimates"] = {k: {kk: vv for kk, vv in (H[k] or {}).items() if kk in ("stat", "p_iut", "p", "ci95", "lower95_one_sided")} for k in H}
        out["models"][model] = entry
    return out


def feature_manifest(out_path=None):
    """Freeze F3: the sha256 of every feature file, committed before any test runner reads TEST features."""
    out_path = Path(out_path or RES / "feature_manifest.json")
    man = {}
    for p in sorted(FEAT.rglob("*")):
        if p.is_file() and p.name != "progress.json":
            man[str(p.relative_to(FEAT))] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    write_json({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "files": man}, out_path)
    return man


# =============================================================================================== cli
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["dev-freeze", "t1", "t2", "t3", "t4", "t5", "t6-prepare", "t6", "cost", "primary", "manifest"])
    a = ap.parse_args(argv)
    if a.what == "dev-freeze":
        dev_freeze()
        return
    if a.what == "t6-prepare":
        t6_prepare()
        return
    if a.what == "manifest":
        feature_manifest()
        return
    if a.what in ("t1", "t2", "t3", "t4", "t5", "t6") and not (RES / "feature_manifest.json").exists():
        raise SystemExit("freeze F3 first: python prereg/analysis.py manifest, then commit and push it")
    freeze = json.load(open(PRE / "dev_freeze.json")) if a.what not in ("t6", "cost", "primary") else {}
    fn = {"t1": lambda: run_t1(freeze), "t2": lambda: run_t2(freeze), "t3": lambda: run_t3(freeze), "t4": lambda: run_t4(freeze),
          "t5": lambda: run_t5(freeze), "t6": run_t6, "cost": run_cost, "primary": run_primary}[a.what]
    t0 = time.time()
    res = fn()
    res["analysis_seconds"] = time.time() - t0
    path = RES / f"{a.what}.json"
    first = not path.exists()
    write_json(res, path)
    with open(RES / "run_log.jsonl", "a") as f:
        f.write(json.dumps({"what": a.what, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "first_run": first, "sha256": sha256_file(path)}) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
