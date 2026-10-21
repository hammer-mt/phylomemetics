# Translation protocol (T1c, and the T2 Chinese secondary), v2

## Language and why

The target language is Simplified Chinese. The scout found three reasons for it: it is the cheapest language in tokens for both models (0.96 times English for Qwen, 1.28 times for Llama); it shares no script with English, so the English sources and the translated targets have no lexical overlap; and both models understood made-up Chinese stories (5 of 6 detail questions each). Chinese is not one of Llama 3.1's supported languages, so a weaker Llama result is expected. No other language is run.

T1c's agent translations are **machine-translated Chinese**: sentence-aligned translations of English summaries. A natural arm (below) uses Chinese text written independently on zh.wikipedia.

## What is translated

- **T1c.** The name-free texts of the 20 DEV Batman screen works, the 7 fresh adaptations and the 69 DEV reference films. The 12 comics stay in English. The translated reference films set the presence bars of the Chinese texts; the comics keep the English bars (language-matched bars).
- **T2.** The banned and control versions of every valid planted plot (272 stories).

## Step 0: Chinese ban lists (T2)

Each meme's ban list is translated into Chinese by two translators, a fresh Claude session and the cross-family check reader, each asked for every common Chinese rendering of each term. The union is the meme's Chinese ban list (`prereg/agent/translation/ban_lists_zh.json`).

## Step 1: translate (agent)

Each call goes to a fresh session that translates one work:

```
Translate this plot summary into Simplified Chinese, as it would read in a Chinese-language plot summary.
Rules:
1. Translate sentence by sentence: the output has exactly the same number of sentences, and Chinese
   sentence k translates English sentence k only.
2. Codes such as X1 stand for names: copy every code exactly as it is, in the same sentence.
3. Invented personal names, if any, are transliterated into Chinese characters, the same way every time.
4. Translate faithfully: add nothing, leave nothing out, do not explain, simplify or improve. Keep the
   wording's level of explicitness: if the English only shows an event, do not name it; if the English
   avoids a usual word, do not use the usual Chinese word for it either.
5. Never use the standard Chinese name of any famous character, place or franchise (for example 蝙蝠侠,
   哥谭, 韦恩, 罗宾, 小丑), even where the English describes one.
6. Use Chinese characters throughout; no English words except the codes.
Return JSON only: {"id": "{WORK_ID}", "sentences": [...]}
<numbered English sentences>
```

## Step 2: back-translate (cross-family)

The cross-family check reader (Gemma 2 9B, `readers.py check --task backtranslate`) translates the Chinese back into English sentence by sentence. A different model family is used so that a Claude back-translator cannot rebuild a known plot from memory.

## Step 3: compare (agent)

A fresh Claude session sees each original English sentence beside its back-translation and answers `same`, `partly` or `different` in meaning.

## Checks

- **L1 alignment (script, `analysis.zh_checks`).** Equal sentence counts; every X-code preserved in its sentence and none added; CJK in every sentence; no Latin word except the codes.
- **L2 meaning (comparison).** At least 90% of pairs `same`, none `different`.
- **L3 names and words (script).** No standard franchise rendering (`analysis.FRANCHISE_ZH`); for T2 banned and control stories, none of the meme's Chinese ban-list terms.

A work that fails is translated again once by a fresh session with the failure list appended; a work that still fails is kept and flagged, and T1c is also reported without flagged works.

## Natural arm (T1c, secondary)

- **N1 fetch (script, after F0).** For each of the 20 DEV Batman screen works: the zh.wikipedia article linked from the English article's Wikidata item (zhwiki sitelink); its first section titled 剧情, 劇情, 情节, 情節, 故事 or 概要; sentences split at 。！？; kept when at least 8 sentences; capped at 40. Fetched on one date, saved raw to `prereg/agent/zhwiki/plots.jsonl` with the date, and committed with freeze F2-T1c.
- **N2 name removal (agent).** The name-removal rewrite prompt of `name_removal_protocol.md`, translated into Chinese, then `analysis.neutralise_cjk` (character-level diff) and the N3 audit. Output `prereg/agent/zhwiki/rewrites.jsonl`.
- `build_lists.py texts zhwiki` assembles `t1c_zhwiki`. The rows present are analysed against the English comics with the translated reference films' bars.

## Outputs

Translations go to `prereg/agent/translation/zh.jsonl` as `{"id", "sentences", "flags"}`. `build_lists.py texts zh` assembles `t1c_zh`, `t1c_fresh_zh` and `t2_stories_zh`; `build_lists.py texts codes` adds the code-collapsed copies for the X-code sensitivity.

## How the methods see Chinese

- **Neurons.** Run on the Chinese texts as on any text; bars from the translated reference films for Chinese rows and from the English name-free films for the English comics.
- **TF-IDF, word presence, LSA and LDA.** Character uni- and bigrams when the corpus contains CJK. Their overlap with the English comics is near zero by construction (a construction check, not a prediction).
- **Keyword queries (T2).** Each English query (plain and evasion-aware) is translated by Qwen3-8B (`readers.py zh-keywords`) and matched as Chinese substrings.
- **Multilingual embeddings.** Qwen3-Embedding, e5 and LaBSE run directly; MiniLM is English-only and is run and reported anyway.
- **LLM readers.** They read the Chinese texts with the same English prompts and summarise in English (the prompts say so; summaries containing CJK are counted and reported).
- **Machine translation plus words (secondary).** Qwen3-8B translates the Chinese back into English (`readers.py mt`), and TF-IDF, words and LSA are fitted on the result. Its cost is counted as LLM reading.
