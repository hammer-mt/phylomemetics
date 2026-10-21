"""Write report/vision.js (the page's data) from results/vision_final.json."""
import json
from pathlib import Path

Path("report/vision.js").write_text("window.PM_VISION = " + json.dumps(json.load(open("results/vision_final.json"))) + ";\n")
print("wrote report/vision.js")
