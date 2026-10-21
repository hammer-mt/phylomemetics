"""Third language probe, MADE-UP text only (no corpus or TEST text; no activations read).
(a) Token cost, decode round trip and extract.py span mapping for Spanish, German, French and Thai on the same three
    short sentences used in lang_check.py (Llama 3.1's official languages include es/de/fr/th/hi but not zh/ja/ru).
(b) Comprehension of a 24-sentence made-up story (llm_timing.STORY) in English, Chinese and Spanish: six
    detail questions answered in English. Translations written by Claude, as the TEST translations would be.
usage: HF_HUB_OFFLINE=1 python lang_check3.py <tag>"""
import json
import sys

sys.path.insert(0, "<repo>")
sys.path.insert(0, "<scratchpad>/prereg/scripts")
from mlx_lm import load, generate  # noqa: E402
from extract import MODELS, sentence_spans  # noqa: E402
from lang_check import S, chat  # noqa: E402
from llm_timing import STORY  # noqa: E402

OUT = "<scratchpad>/prereg/scout/lang_check3_{}.json"
S2 = {
    "es": ["Después de que la hija del farero desaparece durante una tormenta, su hermano distanciado regresa a la isla y descubre que el capitán del puerto ha estado falsificando los registros de los barcos.",
           "X1 esconde las cartas robadas en la capilla, pero X3 la sigue hasta allí y amenaza con contárselo al obispo.",
           "El viejo pescador se niega a vender su barco, así que los promotores incendian su taller."],
    "de": ["Nachdem die Tochter des Leuchtturmwärters während eines Sturms verschwindet, kehrt ihr entfremdeter Bruder auf die Insel zurück und erfährt, dass der Hafenmeister die Schiffsregister gefälscht hat.",
           "X1 versteckt die gestohlenen Briefe in der Kapelle, aber X3 folgt ihr dorthin und droht, es dem Bischof zu erzählen.",
           "Der alte Fischer weigert sich, sein Boot zu verkaufen, also zünden die Bauträger seine Werkstatt an."],
    "fr": ["Après la disparition de la fille du gardien de phare pendant une tempête, son frère, avec qui elle était brouillée, revient sur l'île et apprend que le capitaine du port falsifie les registres des navires.",
           "X1 cache les lettres volées dans la chapelle, mais X3 la suit jusque-là et menace de tout dire à l'évêque.",
           "Le vieux pêcheur refuse de vendre son bateau, alors les promoteurs mettent le feu à son atelier."],
    "th": ["หลังจากลูกสาวของผู้ดูแลประภาคารหายตัวไประหว่างพายุ พี่ชายที่ห่างเหินกันกลับมาที่เกาะและพบว่านายท่าเรือปลอมแปลงบันทึกของเรือมาตลอด",
           "X1 ซ่อนจดหมายที่ถูกขโมยไว้ในโบสถ์ แต่ X3 ตามเธอไปที่นั่นและขู่ว่าจะบอกบิชอป",
           "ชาวประมงชราไม่ยอมขายเรือของเขา นักพัฒนาที่ดินจึงจุดไฟเผาโรงงานของเขา"],
}
ZH = ["玻璃匠伊尔莎的母亲在春季洪水中溺亡后，她继承了运河岛上一间濒临倒闭的作坊。",
      "由一位笑容可掬的港口书记员领导的岛议会，想要把作坊的熔炉拿去建新的海关大楼。",
      "伊尔莎发现母亲留下的账本记录着一位盐商付给书记员的贿赂。",
      "她把账本藏进一盏破裂的枝形吊灯里，谁也没告诉，连她的学徒托马斯也不知道。",
      "夜里，她开始吹制彩色玻璃浮球，并把它们放进水中漂流，向渔民传递暗号警告。",
      "渔民们开始避开盐商的渔网，他的运盐驳船在码头上渐渐腐烂。",
      "书记员怀疑是学徒所为，便让两名码头工人在钟楼后的小巷里把他痛打了一顿。",
      "伊尔莎照料托马斯养伤，心怀愧疚，终于把账本给他看了。",
      "托马斯认出账本上的一个签名属于他自己的叔叔——岛上的地方法官。",
      "两人大吵一架；托马斯离开作坊，搬去和叔叔同住。",
      "一场冬季风暴封锁了港口，岛上的灯油和面包都短缺了。",
      "盐商提出，谁愿意签字让出自己船只的泊位，就给谁粮食。",
      "伊尔莎熔掉母亲留下的最后一盏吊灯，为灯塔制作透镜，因为灯塔的灯已经坏了。",
      "有了新透镜，灯塔引导一艘救援船穿过风暴，岛民们欢呼的是灯塔看守人，而不是她。",
      "托马斯心中不安，搜查了叔叔的书房，发现了第二本账本，上面写着溺亡母亲的名字，旁边记着一笔款项。",
      "他意识到，她溺亡的那天夜里，防洪闸门是被故意打开的。",
      "他把第二本账本带给伊尔莎，两人在结冰的运河上和好。",
      "在春季庆典上，伊尔莎揭开一个巨大的玻璃面具，把账本的内容投射到市政厅的墙上。",
      "人群默默地读着那些名字，盐商试图乘运盐驳船逃走。",
      "那艘驳船在码头上搁了一冬，早已腐朽，在港口入口处散了架。",
      "地方法官辞职，并供认他签字只是为了保住侄子的学徒身份。",
      "书记员被流放到大陆，海关大楼也始终没有建成。",
      "伊尔莎让托马斯成为她的合伙人而不再是学徒，两人一起重建了熔炉。",
      "在洪水周年纪念日，他们把一个透明的玻璃浮球放进水里，里面没有任何讯息。"]
ES = ["Una sopladora de vidrio llamada Ilse hereda un taller en quiebra en una isla de canales después de que su madre se ahoga en la crecida de primavera.",
      "El consejo de la isla, dirigido por un sonriente escribano del puerto, quiere el horno del taller para una nueva aduana.",
      "Ilse descubre que su madre llevaba libros de cuentas que registraban los sobornos que un comerciante de sal pagaba al escribano.",
      "Esconde los libros dentro de una lámpara de araña agrietada y no se lo cuenta a nadie, ni siquiera a su aprendiz Tomas.",
      "Por las noches empieza a soplar flotadores de vidrio de colores y los deja a la deriva con advertencias en clave para los pescadores.",
      "Los pescadores empiezan a evitar las redes del comerciante, y sus barcazas de sal se pudren en el muelle.",
      "El escribano sospecha del aprendiz y hace que dos estibadores le den una paliza en un callejón detrás del campanario.",
      "Ilse cuida de Tomas y, sintiéndose culpable, por fin le enseña los libros de cuentas.",
      "Tomas reconoce en los libros una firma que pertenece a su propio tío, el magistrado de la isla.",
      "Los dos discuten; Tomas abandona el taller y se va a vivir con su tío.",
      "Una tormenta de invierno cierra el puerto, y en la isla escasean el aceite para lámparas y el pan.",
      "El comerciante ofrece grano a quien firme la cesión de los amarres de su barco.",
      "Ilse funde la última lámpara de araña de su madre para hacer lentes para el faro, cuya lámpara se ha averiado.",
      "Con las nuevas lentes, el faro guía a un barco de socorro a través de la tormenta, y la isla aclama al farero, no a ella.",
      "Tomas, inquieto, registra el despacho de su tío y encuentra un segundo libro de cuentas en el que figura el nombre de la madre ahogada con un pago al lado.",
      "Se da cuenta de que las compuertas se dejaron abiertas a propósito la noche en que ella se ahogó.",
      "Lleva el segundo libro a Ilse, y se reconcilian sobre el canal helado.",
      "En la fiesta de primavera, Ilse descubre una gran máscara de vidrio que proyecta las páginas del libro de cuentas sobre la pared del ayuntamiento.",
      "La multitud lee los nombres en silencio, y el comerciante intenta huir en una barcaza de sal.",
      "La barcaza, podrida tras un invierno en el muelle, se parte en la bocana del puerto.",
      "El magistrado dimite y confiesa que solo firmó para proteger el aprendizaje de su sobrino.",
      "El escribano es desterrado al continente, y la aduana nunca llega a construirse.",
      "Ilse toma a Tomas como socio en lugar de aprendiz, y juntos reconstruyen el horno.",
      "En el aniversario de la crecida, dejan a la deriva un flotador de vidrio transparente sin ningún mensaje dentro."]
QS = ["Where did Ilse hide the ledgers?", "Who had the apprentice beaten, and where?", "Whose signature did Tomas recognise in the ledgers?",
      "What did the second ledger reveal about the mother's death?", "What did Ilse make from her mother's last chandelier, and why?",
      "How did the merchant try to escape, and what happened?"]
GOLD = ["inside a cracked chandelier", "the harbour clerk; two dockhands; alley behind the bell tower", "his uncle, the magistrate",
        "the flood gates were opened on purpose (a payment beside her name): she was killed", "lenses for the lighthouse, whose lamp had failed",
        "on a salt barge, which broke apart in the harbour mouth"]
NAMES = {"en": "English", "zh": "Chinese", "es": "Spanish"}


def main(tag):
    model, tok = load(MODELS[tag])
    hf = tok._tokenizer
    res = {"model": MODELS[tag], "tokens": {}, "roundtrip": {}, "spans_ok": {}, "story": {}, "gold": GOLD}
    en_tok = sum(len(hf(s, add_special_tokens=False)["input_ids"]) for s in S["en"])
    for lang, ss in {**S2, "en": S["en"], "zh": S["zh"]}.items():
        n = [len(hf(s, add_special_tokens=False)["input_ids"]) for s in ss]
        res["tokens"][lang] = {"ratio_to_en": round(sum(n) / en_tok, 2)}
        res["roundtrip"][lang] = all(hf.decode(hf(s, add_special_tokens=False)["input_ids"]) == s for s in ss)
        ids, spans = sentence_spans(hf, ss)
        res["spans_ok"][lang] = bool(all(e > s for s, e in spans) and all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1)))
    for lang, ss in {"en": STORY, "zh": ZH, "es": ES}.items():
        n = sum(len(hf(s, add_special_tokens=False)["input_ids"]) for s in ss)
        ids, spans = sentence_spans(hf, ss)
        p = " ".join(ss) if lang != "zh" else "".join(ss)
        qs = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(QS))
        a = generate(model, tok, chat(tok, f"Story ({NAMES[lang]}):\n\n{p}\n\nAnswer in English, one short line each:\n{qs}"), max_tokens=260, verbose=False).strip()
        res["story"][lang] = dict(tokens=n, spans_ok=len(spans) == len(ss), answers=a)
        print(lang, n, a, flush=True)
    json.dump(res, open(OUT.format(tag), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: res[k] for k in ("tokens", "roundtrip", "spans_ok")}, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
