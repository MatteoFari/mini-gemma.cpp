"""Compare repeated shared-context requests in one process per cache setting."""
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
executable = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else root / "build/inference_engine"
model = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else root / "models/gemma-4-E2B_q4_0-it.gguf"
document = """Answer questions about this project document using a short sentence.
The library project is based in Turin. Its maintainer is Elena. The nightly backup starts at 21:00.
The team keeps a catalog of books, journals, and historical maps. Every item has a stable identifier,
a title, an author, and a shelf location. Readers can search the catalog or request help at the desk.
Returned books are checked for damage before staff put them back on the shelves. Rare maps stay in
a separate room with controlled lighting. Staff record every loan and return in the local database.
The catalog is exported each evening before the backup begins. A second copy is kept off site.
Visitors can reserve a reading desk for two hours. Reservations are confirmed at the front desk.
The library hosts workshops on finding reliable sources and preserving family records. Volunteers
help digitize public records, check scanned pages, and correct catalog entries. New volunteers
complete a training session before handling fragile material. The project document is reviewed
every month so that the opening schedule and contact details remain accurate. Elena reviews the
changes and publishes the approved version. The library project remains based in Turin."""
questions = ["Which city is the project based in?", "Who is the maintainer?", "When does the backup start?",
             "How long can a visitor reserve a desk?", "Where are rare maps stored?", "What do volunteers do?"]
requests = ["Name the project maintainer."] + questions  # One warm-up, then six different questions.
session = "".join(f"{q}\n/metrics\n/reset\n" for q in requests) + "/exit\n"
results, replies = {}, {}
for mib in (0, 128):
    command = [str(executable), str(model), "--system", document, "--max-tokens", "24",
               "--threads", "4", "--prefix-cache-mib", str(mib)]
    run = subprocess.run(command, input=session, text=True, capture_output=True, timeout=300, check=True)
    if "Error:" in run.stdout:
        raise RuntimeError(run.stdout)
    samples = re.findall(r"Prompt: (\d+) tokens \| prefill: ([\d.]+) s \| reused: (\d+) tokens", run.stdout)
    replies[mib] = re.findall(r"AI > (.*?)\nUser >", run.stdout, re.S)
    if len(samples) != len(requests) or len(replies[mib]) != len(requests):
        raise RuntimeError("Incomplete benchmark output")
    results[str(mib)] = [{"question": q, "prompt_tokens": int(n), "prefill_seconds": float(t),
                          "reused_tokens": int(r)} for q, (n, t, r) in zip(questions, samples[1:])]
if replies[0] != replies[128]:
    raise AssertionError("Cached and uncached generated replies differ")
medians = {key: statistics.median(row["prefill_seconds"] for row in rows) for key, rows in results.items()}
print(json.dumps({"threads": 4, "samples": results, "median_prefill_seconds": medians,
                  "prefill_speedup": medians["0"] / medians["128"], "identical_replies": True}, indent=2))
