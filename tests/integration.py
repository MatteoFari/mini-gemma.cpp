"""Real-model tests, golden data comes from the independent pinned llama.cpp oracle."""
import array
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

executable, model = map(str, map(Path, sys.argv[1:3]))
base = [executable, model, "--max-tokens", "24"]
checks = 0

def run(*arguments, stdin=None, success=True):
    result = subprocess.run(base + list(arguments), input=stdin, capture_output=True, text=True, timeout=300)
    expected = 0 if success else 1
    if result.returncode != expected:
        raise AssertionError(f"Exit {result.returncode}, expected {expected}: {result.stderr}")
    return result

def check(condition, message):
    global checks
    if not condition:
        raise AssertionError(message)
    checks += 1

with tempfile.TemporaryDirectory() as directory:
    golden = json.loads(Path(__file__).with_name("golden.json").read_text())
    for case in golden:
        prompt = case.get("text", " a" * case.get("repetitions", 0))
        ids = list(map(int, run("--tokens", prompt).stdout.split()))
        expected_ids = case.get("ids", [2] + [496] * case.get("repetitions", 0))
        check(ids == expected_ids, "Tokenizer differs from independent oracle")
        output = Path(directory) / "logits.f32"
        run("--prompt", prompt, "--logits", str(output))
        scores = array.array("f")
        scores.frombytes(output.read_bytes())
        check(len(scores) == 262144 and all(map(math.isfinite, scores)), "Invalid vocabulary score vector")
        check(max(range(len(scores)), key=scores.__getitem__) == case["top"], "Reference top token changed")
        check(max(abs(scores[i] - expected) for i, expected in case["scores"]) < case["tolerance"],
              "Reference score regression")
    bad = Path(directory) / "truncated.gguf"
    bad.write_bytes(b"GGUF" + b"\0" * 28)
    result = subprocess.run([executable, str(bad), "--prompt", "Hello"], capture_output=True, text=True)
    check(result.returncode == 1, "Malformed model should fail cleanly")

english = "What is the capital of France? Answer in one short sentence."
answer = run("--prompt", english)
check(answer.stdout.strip() == "The capital of France is Paris.", "English generation failed")
italian = "Qual è la capitale della Francia? Rispondi in italiano con una sola frase."
italian_answer = run("--prompt", italian).stdout.strip()
check("Parigi" in italian_answer, "Italian generation failed")

print(f"{checks} reference and generation assertions passed.")
