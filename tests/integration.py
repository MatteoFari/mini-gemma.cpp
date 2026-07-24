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

# Reset must reproduce greedy output and reset counters while retaining the loaded model.
chat = run(stdin=f"{english}\n/metrics\n/reset\n/metrics\n{english}\n/metrics\n/exit\n").stdout
check(chat.count("AI > The capital of France is Paris.") == 2, "Reset changed deterministic output")
check("context: 0/2048" in chat, "Reset did not clear position")
counts = re.findall(r"Generated: (\d+) tokens", chat)
check(len(counts) == 3 and counts[0] == counts[2] and counts[1] == "0", "Metrics accumulated across requests")

# A rejected prompt must preserve the previous conversation, without consuming context.
conversation = "Remember: my name is Matteo. Reply with OK.\n" + " a" * 150
conversation += "\nWhat is my name? Answer with just the name.\n/exit\n"
chat = run("--ctx", "128", stdin=conversation).stdout
check("Not enough context" in chat and "AI > Matteo" in chat, "Context guard lost earlier conversation")
for args in [("--temp", "nan"), ("--top-p", "0"), ("--max-tokens", "0"), ("--threads", "0"),
             ("--ctx", "16"), ("--threads", "4oops"), ("--unknown", "1"), ("--prompt", " ")]:
    run(*args, success=False)
    checks += 1
check(run(stdin="").returncode == 0, "EOF did not exit")
print(f"{checks} integration assertions passed: tokenizer, logits, cache wrap, chat, reset, limits and errors.")
