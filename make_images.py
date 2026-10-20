"""Generate the page's original illustrations and the vision-probe images with Gemini.

No Batman likeness, logos or text: the memes are shown as objects and scenes. Saved to report/img/ (page)
and data/probes/ (images the vision neurons are tested on). Skips files that already exist.
"""
import os
import sys
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

STYLE = ("Ink-and-wash editorial illustration in the manner of a 19th-century natural-history engraving: fine cross-hatched "
         "linework on off-white paper, muted grey-blue tones with a single warm yellow accent. No text, no letters, no logos, "
         "no superhero costumes, no recognisable characters or real people.")
FILM = ("Cinematic film still, moody natural light, 35mm. No text, no logos, no symbols, no recognisable characters or real people, "
        "no face visible.")

JOBS = {
    "report/img/hero.jpg": ("16:9", STYLE + " A natural-history specimen plate: eight objects arranged in a neat grid like pinned museum "
                            "specimens, each casting a soft shadow: a black domino mask, a tarnished silver coin with one face scratched, "
                            "a bat with its wings spread, a broken string of pearls, a small glass flask of green liquid, a playing card "
                            "face down, a butler's silver hand bell, and a grappling hook with coiled rope."),
    "report/img/meme_origin.jpg": ("4:3", STYLE + " A broken pearl necklace scattered across wet cobblestones in a dark alley at night, lit by one streetlamp."),
    "report/img/meme_identity.jpg": ("4:3", STYLE + " A black domino mask resting on an antique mahogany desk beside reading glasses and a stack of business papers, in a wealthy study."),
    "report/img/meme_butler.jpg": ("4:3", STYLE + " An elderly butler's white-gloved hands holding a silver tray with a first-aid kit and a cup of tea, in a grand old mansion hallway."),
    "report/img/meme_coin.jpg": ("4:3", STYLE + " A silver coin spinning in mid-air above an open palm, one face deeply scarred and scratched."),
    "report/img/meme_vat.jpg": ("4:3", STYLE + " An industrial chemical plant at night: a metal catwalk above a huge vat of glowing green liquid, with steam rising."),
    "report/img/meme_funeral.jpg": ("4:3", STYLE + " Mourners with black umbrellas around an open grave in the rain, a small boy standing a little apart."),
    "report/img/meme_bats.jpg": ("4:3", STYLE + " A swarm of bats bursting from the mouth of a dark cave at dusk."),
    "data/probes/cape.jpg": ("1:1", FILM + " A caped figure seen from behind, silhouetted on a gothic rooftop beside a stone gargoyle, looking over a rainy city at night."),
    "data/probes/skyline.jpg": ("1:1", FILM + " A gothic art-deco city skyline at night under heavy clouds, searchlights sweeping the sky."),
    "data/probes/bats.jpg": ("1:1", FILM + " A swarm of bats bursting out of a cave mouth at dusk."),
    "data/probes/mansion.jpg": ("1:1", FILM + " A grand, gloomy old mansion on a hill at night, one window lit."),
    "data/probes/beach.jpg": ("1:1", FILM + " A sunny beach in bright daylight with umbrellas and a lifeguard tower."),
    "data/probes/kitchen.jpg": ("1:1", FILM + " A cosy farmhouse kitchen in morning light, fresh bread on a wooden table."),
    "data/probes/meadow.jpg": ("1:1", FILM + " A flower meadow with grazing horses on a clear spring afternoon."),
}


def main():
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for path, (ratio, prompt) in JOBS.items():
        p = Path(path)
        if p.exists():
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                r = client.models.generate_content(
                    model="gemini-3-pro-image-preview", contents=[prompt],
                    config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"],
                                                       image_config=types.ImageConfig(aspect_ratio=ratio, image_size="1K")))
                img = next(part.as_image() for part in r.parts if part.inline_data)
                img.save(str(p))
                im = Image.open(p).convert("RGB")
                if p.parts[0] == "report":
                    im.thumbnail((1100, 1100))
                im.save(p, "JPEG", quality=82)
                print("made", path, flush=True)
                break
            except Exception as e:  # noqa: BLE001
                print("retry", path, e, file=sys.stderr, flush=True)
        else:
            print("FAILED", path, flush=True)


if __name__ == "__main__":
    main()
