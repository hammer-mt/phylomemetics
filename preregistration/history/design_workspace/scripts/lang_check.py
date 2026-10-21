"""E4 feasibility check on made-up (non-corpus) plot sentences only. For each model: tokenisation cost per
language, decode round trip, extract.py's sentence-span mapping, back-translation into English, and a short
continuation in the target language. No neuron activations are read.

usage: HF_HUB_OFFLINE=1 python lang_check.py <qwen3-8b|llama3.1-8b>"""
import json
import sys
import time

sys.path.insert(0, "<repo>")
from mlx_lm import load, generate  # noqa: E402
from extract import MODELS, sentence_spans  # noqa: E402

OUT = "<scratchpad>/prereg/scout/lang_check_{}.json"

S = {
    "en": ["After the lighthouse keeper's daughter vanishes during a storm, her estranged brother returns to the island and learns that the harbour master has been forging the ship records.",
           "X1 hides the stolen letters in the chapel, but X3 follows her there and threatens to tell the bishop.",
           "The old fisherman refuses to sell his boat, so the developers set fire to his workshop."],
    "zh": ["灯塔看守人的女儿在一场暴风雨中失踪后，她疏远多年的哥哥回到岛上，得知港务长一直在伪造船只记录。",
           "X1把偷来的信件藏在小教堂里，但X3跟踪她到那里，并威胁要告诉主教。",
           "老渔夫拒绝卖掉他的船，于是开发商放火烧了他的作坊。"],
    "ja": ["嵐の最中に灯台守の娘が行方不明になった後、疎遠になっていた兄が島に戻り、港長が船の記録を偽造していたことを知る。",
           "X1は盗んだ手紙を礼拝堂に隠すが、X3は彼女の後をつけてそこへ行き、司教に告げると脅す。",
           "年老いた漁師が船を売ることを拒んだため、開発業者たちは彼の作業場に火を放つ。"],
    "ru": ["После того как дочь смотрителя маяка пропадает во время шторма, её давно отдалившийся брат возвращается на остров и узнаёт, что начальник порта подделывал судовые записи.",
           "X1 прячет украденные письма в часовне, но X3 следует за ней туда и угрожает рассказать епископу.",
           "Старый рыбак отказывается продавать свою лодку, и тогда застройщики поджигают его мастерскую."],
    "hi": ["तूफ़ान के दौरान प्रकाशस्तंभ के रखवाले की बेटी के लापता हो जाने के बाद, उसका बिछड़ा हुआ भाई द्वीप पर लौटता है और उसे पता चलता है कि बंदरगाह अधिकारी जहाज़ों के रिकॉर्ड में जालसाज़ी करता रहा है।",
           "X1 चुराए गए पत्रों को गिरजाघर में छिपा देती है, लेकिन X3 उसका पीछा करते हुए वहाँ पहुँच जाता है और बिशप को बताने की धमकी देता है।",
           "बूढ़ा मछुआरा अपनी नाव बेचने से इनकार कर देता है, इसलिए बिल्डर उसकी कार्यशाला में आग लगा देते हैं।"],
}
CONTINUE = {"zh": "请用中文为下面这段情节续写两句话，只输出续写内容：\n\n",
            "ja": "次のあらすじの続きを日本語で二文だけ書いてください。続きだけを出力してください：\n\n",
            "ru": "Продолжи этот сюжет двумя предложениями на русском языке. Выведи только продолжение:\n\n",
            "hi": "इस कहानी को हिंदी में केवल दो वाक्यों में आगे बढ़ाइए। केवल आगे की कहानी लिखिए:\n\n"}


def chat(tok, text):
    msgs = [{"role": "user", "content": text}]
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def main(tag):
    model, tok = load(MODELS[tag])
    hf = tok._tokenizer
    res = {"model": MODELS[tag], "tokens": {}, "roundtrip": {}, "spans_ok": {}, "back_translation": {}, "continuation": {}}
    en_tok = [len(hf(s, add_special_tokens=False)["input_ids"]) for s in S["en"]]
    for lang, ss in S.items():
        n = [len(hf(s, add_special_tokens=False)["input_ids"]) for s in ss]
        res["tokens"][lang] = {"per_sentence": n, "ratio_to_en": round(sum(n) / sum(en_tok), 2),
                               "chars_per_token": round(sum(map(len, ss)) / sum(n), 2)}
        res["roundtrip"][lang] = all(hf.decode(hf(s, add_special_tokens=False)["input_ids"]) == s for s in ss)
        ids, spans = sentence_spans(hf, ss)
        ok = all(e > s for s, e in spans) and all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))
        res["spans_ok"][lang] = {"ok": bool(ok), "spans": spans, "n_ids": len(ids)}
        # how X-codes tokenise inside the sentence
        if lang != "en":
            x = hf(ss[1], add_special_tokens=False)["input_ids"][:6]
            res["spans_ok"][lang]["xcode_head_tokens"] = [hf.decode([i]) for i in x]
            t0 = time.time()
            res["back_translation"][lang] = generate(model, tok, chat(tok, "Translate the following text into English. Output only the translation.\n\n" + " ".join(ss)),
                                                     max_tokens=160, verbose=False).strip()
            res["continuation"][lang] = generate(model, tok, chat(tok, CONTINUE[lang] + ss[0]), max_tokens=140, verbose=False).strip()
            print(lang, f"{time.time() - t0:.1f}s", flush=True)
    json.dump(res, open(OUT.format(tag), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
