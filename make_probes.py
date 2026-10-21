"""Generate the probe images the vision neurons are tested on (data/probes/) with Gemini.

No Batman likeness, logos or text. Skips files that already exist. Needs GEMINI_API_KEY.
"""
import os
import sys
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

FILM = ("Cinematic film still, moody natural light, 35mm. No text, no logos, no symbols, no recognisable characters or real people, "
        "no face visible.")

JOBS = {
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
