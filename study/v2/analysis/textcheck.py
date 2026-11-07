"""Word matching for the explanation tests (PLAN.md, Section 10): stems, and whether a text uses a key word.

A key word "is used" in a text when the stem of any of its content words equals the stem of a word of the text.
Stems are crude on purpose (suffixes stripped), so that inflections and derivatives count: revenge, revenged,
avenging and avenger all reduce to one of two stems. Function words inside a multi-word key term do not count.
"""
import re

STOP = set("a an and are as at be been but by for from had has have he her his if in into is it its of on or she so than that the their them "
           "then there these they this to was were which while who whom whose will with would not no nor do does did over under about after "
           "before between during up down out off own same such too very can just one two also".split())
WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
SUFFIXES = ("ingly", "edly", "ations", "ation", "isms", "ism", "ists", "ist", "ments", "ment", "ness", "ings", "ing", "ies", "ied", "ers", "est",
            "ed", "es", "er", "ly", "s")


def stem(word):
    w = word.lower()
    if w.endswith("'s"):
        w = w[:-2]
    for _ in range(3):                                   # murdered -> murder -> murd, as murders -> murd
        before = w
        for suf in SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[:-len(suf)] + ("y" if suf in ("ies", "ied") else "")
                break
        if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiouls":      # stopped -> stop
            w = w[:-1]
        if len(w) > 3 and w.endswith("e"):
            w = w[:-1]
        if len(w) > 4 and w.endswith("y"):                                # wealthy -> wealth
            w = w[:-1]
        if w == before:
            break
    return w


def stems(text):
    return {stem(w) for w in WORD.findall(text.lower())}


def key_stems(terms):
    """Stems of the content words of a list of key words or phrases."""
    out = set()
    for t in terms:
        out |= {stem(w) for w in WORD.findall(str(t).lower()) if w not in STOP and len(w) > 1}
    return out


def uses(text, key):
    """The key stems that the text uses."""
    return stems(text) & key


def phrase_hits(text, terms):
    """How many search terms (words or phrases) occur in the text: every content word of the term must occur, by stem."""
    have = stems(text)
    n = 0
    for t in terms:
        ws = [stem(w) for w in WORD.findall(str(t).lower()) if w not in STOP and len(w) > 1]
        n += bool(ws) and all(w in have for w in ws)
    return n
