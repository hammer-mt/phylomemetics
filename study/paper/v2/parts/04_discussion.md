## 4. Discussion

### 4.1 What the better method changed

Version 1's limitations section, its robustness review and use of the results since publication pointed at its method: small quantised models, a presence rule that counted weak activations, labels read off the corpus, and no causal test. This version addressed each of those, following the essay as far as it publishes its method. The conclusions about Batman did not improve. Where they moved, they moved against the idea that neurons give a better account of a story's lineage than words do.

- Heredity is supported in all four models (in Llama 3.1 8B with a failed positive control), and word counts find the sources about as well. TF-IDF's 11% is ahead of Qwen3-32B (15%) and Llama 3.1 8B (14%) and behind Qwen3-8B (8%) and Llama 3.3 70B (6%), and in no model do the units add anything once the word measures are regressed out.
- The one result in version 1 that suggested neurons add something to words, Llama's residual test, came from a test that words alone can pass. Under a test that no word-only measure we tried passes, no model adds anything: not under the new rule in any of the four models, and not under version 1's rule in the three it was run on.
- Without names the canon is at or below chance at telling Batman from other heroes in the three models up to 32B. Version 1 blamed its scoring rule. Given the words' scoring rule, the units do no better than a count of sentences. The 70B model reaches 0.77, short of the bar, and it is the model that names the exact film for every ordinary plot and says Batman for 29 of the 32 Batman works.
- Version 1's name-free core of seven units, which read like a definition of a superhero, was made of weak activations. None is left in Qwen3-8B and one in Qwen3-32B.

Llama 3.3 70B is the exception three times: on identity, on trees, and on version 1's form of the residual test, which it passes under the new rule (p = 0.017) where the other three models fail. It is the model that recognises the summaries most completely. It names the exact film for all 69 ordinary plots (the other models 36 to 54) and says Batman for 29 of the 32 name-free Batman works (the others 16 to 21), so on this corpus we cannot separate what its units detect from what the model remembers.

What did improve is the thing the essay is about, the units themselves. A description written from a unit's strongest examples predicts where it fires in new text about as well as the essay reports for its own neurons, and nearly half of the units still fire when the idea is expressed without its key words. A second model family largely agrees about what kind of thing each unit is. So the definition this paper started from, that a meme is whatever makes a given neuron fire, can be applied and checked unit by unit. That is a real gain over version 1, where a label was a reading of eight sentences.

### 4.2 Why cleaner units did not give a better lineage

Four things stand between a unit that can be described and a unit that is useful for tracing culture, and this study ran into all of them.

**The canon is made of names.** The units that recur across Batman works and are rare elsewhere respond to Batman's names and to comic-book writing. That is a fair description of what makes a text recognisably Batman, and it is also what any reader or word count would use. Take the names away and what is left is the superhero genre, which Batman shares with the films he is being told apart from.

**Removing names leaves a mark.** The name-free texts were meant to show what remains of a story without its names. The units that best distinguish name-free Batman respond to the placeholders themselves, and a count of placeholders separates Batman from the hero films better than any score built from units or words. Any future version of this test needs rewritten stories, not redacted ones. The planned planted-story experiment is built that way.

**Presence is decided where descriptions are weakest.** The essay's claim is about strong activations, and a description is written from the strongest. Presence in a work is decided at 30% of scale, and there the description accounts for about half of the firings. So a grid of which works carry which memes is noisier than the titles on its rows suggest.

**What can be described and what the model uses are different sets of activity.** The essay's working rule is that a neuron's strong activations carry its meaning, and the explanation tests bear that out. But when Batman's canon is switched off, about three quarters of the effect on the model's answer comes from activity below 30% of scale (1.75 of 2.39 in log-odds), which is the activity the presence rule discards. The rule that makes a unit describable throws away signal the model uses. Version 1's rule kept that signal and could not say what it meant. Neither rule gives both.

The corpus sets a ceiling of its own. Wikipedia describes a film and the comic it adapts in much the same words, so word counts already find most of the sources (TF-IDF ranks the source first for six of the nine adaptations) and leave little for a neuron to add.

### 4.3 Can creativity be reverse-engineered?

Not from this evidence. What the study supports is narrower. Inheritance between stories can be measured automatically, from text, with a test that has a known false-positive rate: adaptations carry their sources' material. A language model's neurons are units that can be named, and the names can be tested and can fail. And a few hundred of those units appear to carry much of what lets Qwen3-32B recognise Batman in a summary with no names in it.

What it does not support is that neurons see more of a story's ancestry than its words do, or that there is a name-free essence of Batman for them to find. On 32 Wikipedia summaries the first is close to untestable, because the words already carry the trail. The experiment that could show it, on stories where the wording has been changed on purpose, is designed and written up as a pre-registration, and has not been run.
