# UN-26: Text-chosen units read from pixels in one model

**Area:** Across models · **Status:** design registered 8 November 2026, not run · **Priority:** 3.0 of 5

## Question

Does a unit that responds to a scene described in words respond to the same scene shown as an image with no lettering?

## Prediction

In Qwen3-VL-8B, 30% or more of units that agree across two people's captions of an image (correlation 0.5 or more) also agree between caption and pixels at 0.3 or more on unseen images, against under 5% for mismatched unit pairs.

## What would refute it

Under 10% of caption-reliable units transfer to pixels, or no more than mismatched unit pairs matched on firing rate, or transfer vanishes on images with no detected lettering. Then unit readings are bound to text. 1,800 held-out images give a per-unit correlation standard error near 0.024.

## Data

Crossmodal-3600 (Thapliyal et al. 2022): 3,600 photographs from around the world with human captions in 36 languages, written independently and not translated; captions CC BY 4.0 (17 MB), images 302 MB with an attribution file. Half the images to choose units, half to score. Typographic decoys (a caption rendered as an image) are generated.

## Method

Model: Qwen3-VL-8B at 4 bits, whose language layers have the same shape as the study's Qwen3-8B; first measure whether a unit index gives the same reading of text in both. Read every unit on caption tokens with no image, and on image tokens with a neutral prompt. Per unit: caption-to-caption, caption-to-pixel and caption-to-decoy correlation. Nulls: mismatched unit pairs; shuffled images.

## Cheap rival it must beat

In one task, finding an image's own caption among 1,800: text-chosen units reading pixels against a contrastive image-text model (CLIP or SigLIP), the model's own written caption matched by TF-IDF, its pooled residual stream, and image brightness and size as the analogue of length. CLIP will win retrieval; the transfer share still says whether a text catalogue reads images.

## Premise it tests

P1 mainly, P2 across media. Transfer means units pass the concept-cell test of invariance (one response to a word and a picture), which name detectors fail. No transfer means one catalogue per medium, and the poster experiment cannot be joined to the text study.

## Biology analogue

Concept cells that fire to a face and its written name.

## Cost

M. New extraction code for a vision-language model, then a few hours of compute for 3,600 images and their captions.

## Payoff

Whether one catalogue can read both text and images, whether the study's own Qwen3-8B units carry over to pictures, and a controlled replacement for the poster experiment, which had no cheap rival and no lettering control.

## Main risk

Captions describe concrete scenes, so a pass shows invariance for objects and places, the easy case, and says little about story memes such as revenge or a secret identity. A model trained to caption images may fire word units on pixels simply because it is about to write the word.

## Reviewer note

Switched to Qwen3-VL-8B (checkpoint confirmed; same layer count and widths as Qwen3-8B) so the study's own units can be tested, and put the rivals into one retrieval task. Remaining: units shared between text and image are already reported for such models, and photographs are far from story memes.

## Sources

- Thapliyal et al. 2022 (Crossmodal-3600)
- Goh et al. 2021
- Quian Quiroga et al. 2005
- Schwettmann et al. 2023
