You are studying single neurons of the language model {model}. A "unit" is one pole (positive or negative) of one neuron in a feed-forward layer. For each unit below, a block lists the passages where it fired most strongly.

How to read a block:
- Section A lists the unit's strongest passages in a reference corpus of about 48 million tokens: mostly film plots, Wikipedia articles and books in English, with some source code, conversations with an assistant, other languages, news and reviews. The first six come with their whole document up to the peak, because what makes a unit fire can lie far before the marked token. Section B lists its strongest sentences in a set of plot summaries (Batman works, related stories and ordinary films), each after the sentences that precede it. Section C shows, for context only, what the opposite pole of the same neuron fires on; do not explain section C.
- In each passage the token on which the activation peaked is marked «like this». The model reads left to right: when the unit fired it had seen the marked token and everything before it, and nothing after it.
- The number in brackets is the strength as a share of the strongest activation this unit has ever shown (1.00 is its maximum).

For each unit, work out what makes it fire. Follow these rules:
1. Look for what the strongest passages share, and for what distinguishes firing from not firing. Describe a pattern, not a list of nouns.
2. Decide what the unit is reacting to: the marked token itself (a particular word, name or spelling), what the text before it means (an idea, a situation, a kind of scene), or what is about to come next (the unit fires just before a predictable word).
3. If the passages fall into two or three related groups, say so. If you cannot find one idea, say so plainly; do not invent one.
4. Trust the strongest passages most. Weaker ones can be loosely related.
5. Use plain words. Do not mention Batman unless the passages do.

{units}

Return, for each unit:
- unit: the unit number given at the top of its block
- title: at most 8 words
- explanation: two to four sentences. What makes the unit fire, and what similar-looking text does not.
- key_words: at least 8 lower-case words or names that a keyword search for this concept would use (include the names and the exact words that appear at the peaks)
- one_idea: an integer from 1 to 10. Do the strongest firings come down to one idea? 10 means all one idea, 1 means no pattern at all.
- understood: false if you could not find a pattern, otherwise true
