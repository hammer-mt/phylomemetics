Each item below is the title and short explanation of something that one neuron unit in a language model detects. You have only these descriptions.

For each item decide:

kind: one of
- "character or name": a particular person, character or other proper name
- "place": a particular place, real or fictional
- "object": a particular thing or class of physical things
- "word or token": a particular word, spelling, word-piece or symbol, whatever it means
- "story pattern": something that happens in a story: an event, a situation, a relationship, a motive, a turn of the plot
- "genre or setting": the kind of story or world it is
- "style or format": how the text is written or laid out (register, cast lists, citations, headings)
- "grammar": a grammatical construction or position in a sentence
- "other"

detects: one of
- "name or word": the unit reacts to a particular name, word or string. Text that conveys the same thing without that name or word would not be expected to fire it.
- "idea": the unit reacts to a meaning that can be put in other words. Text that conveys it without any particular word would be expected to fire it.

{items}

Return, for each item: unit (its number), kind, detects.
