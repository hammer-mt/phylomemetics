"""Second comprehension probe (made-up sentences only): instruction after the passage, plus three short
English questions about it. usage: HF_HUB_OFFLINE=1 python lang_check2.py <tag>"""
import json
import sys

sys.path.insert(0, "<repo>")
sys.path.insert(0, "<scratchpad>/prereg/scripts")
from mlx_lm import load, generate  # noqa: E402
from extract import MODELS  # noqa: E402
from lang_check import S, chat  # noqa: E402

NAMES = {"zh": "Chinese", "ja": "Japanese", "ru": "Russian", "hi": "Hindi"}
QS = ["Who went missing, and during what?", "What does X3 threaten to do?", "Why do the developers set fire to the workshop?"]
OUT = "<scratchpad>/prereg/scout/lang_check2_{}.json"


def main(tag):
    model, tok = load(MODELS[tag])
    res = {}
    for lang in NAMES:
        p = " ".join(S[lang])
        tr = generate(model, tok, chat(tok, f"Here is a short passage in {NAMES[lang]}:\n\n{p}\n\nTranslate the passage above into English. Reply in English only."), max_tokens=160, verbose=False).strip()
        qa = generate(model, tok, chat(tok, f"Passage ({NAMES[lang]}):\n\n{p}\n\nAnswer in English, one short line each:\n1. {QS[0]}\n2. {QS[1]}\n3. {QS[2]}"), max_tokens=120, verbose=False).strip()
        res[lang] = {"translation": tr, "qa": qa}
        print(lang, json.dumps(res[lang], ensure_ascii=False, indent=1), flush=True)
    json.dump(res, open(OUT.format(tag), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
