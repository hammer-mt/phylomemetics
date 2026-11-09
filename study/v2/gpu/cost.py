"""What the rented GPUs cost: the provider's billed total, and each machine's hours from its state file.

    cd study && ../.venv/bin/python v2/gpu/cost.py

Reads the state files in v2/gpu/.state/ (only the GPU type, count, hourly rate and start and end times; never the
access tokens) and the account's billing. Writes results/v2/gpu_cost.json. The provider's billing can lag by some
minutes, so the hours-times-rate figure is given beside it.
"""
import datetime
import glob
import json
import time
from pathlib import Path

import runpod_ctl as rc

HERE = Path(__file__).resolve().parent


def main():
    pods = []
    for f in sorted(glob.glob(str(HERE / ".state" / "*.json"))):
        s = json.load(open(f))
        c = s.get("create_response", {})
        start = s["created"]
        end = s.get("terminated")
        hours = ((end or time.time()) - start) / 3600
        rate = float(c.get("cost") or c.get("costPerHr") or 0)
        pods.append({"gpu": s.get("gpu"), "count": s.get("count"), "dollars_per_hour": rate,
                     "started_utc": datetime.datetime.fromtimestamp(start, datetime.timezone.utc).isoformat(timespec="minutes"),
                     "ended_utc": datetime.datetime.fromtimestamp(end, datetime.timezone.utc).isoformat(timespec="minutes") if end else None,
                     "hours": round(hours, 2), "hours_times_rate": round(hours * rate, 2)})
    bill = rc.spend()
    out = {"provider": "Runpod", "budget_dollars": 200, "machines": pods,
           "hours_times_rate_total": round(sum(p["hours_times_rate"] for p in pods), 2),
           "billed_total_dollars": round(bill["totalAmount"], 2),
           "billed_gpu_dollars": round(bill.get("podGpuAmount", 0), 2), "billed_disk_dollars": round(bill.get("podDiskAmount", 0), 2),
           "billing_read_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="minutes"),
           "all_machines_ended": all(p["ended_utc"] for p in pods)}
    before = 72.7                                # billed when paper v2's own runs and its follow-up were finished (8 November, 07:59 UTC)
    out["billed_before_the_two_further_corpora"] = before
    out["billed_for_the_two_further_corpora"] = round(out["billed_total_dollars"] - before, 2)
    out["machines_not_in_the_list"] = ("The machines of 8 November (18:01 to 18:44 and 20:00 to 20:38 UTC; RTX PRO 6000; the pipeline checks before the second plan was "
                                       "frozen, about 7 dollars) are not listed: their state files were reused for later machines. The billed total includes them.")
    dst = HERE.parent.parent / "results" / "v2" / "gpu_cost.json"
    dst.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", dst.relative_to(HERE.parent.parent), "| billed", out["billed_total_dollars"], "| hours x rate", out["hours_times_rate_total"], "|", [(p["gpu"], p["hours"], p["dollars_per_hour"]) for p in pods])


if __name__ == "__main__":
    main()
