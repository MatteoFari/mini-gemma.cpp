"""Download the pinned public Gemma4 E2B GGUF and verify its published SHA-256."""
import hashlib
import urllib.request
import tempfile
from pathlib import Path

revision = "675cff42a74c774d6cb76f76d8eacb49b48c9b93"
name = "gemma-4-E2B_q4_0-it.gguf"
expected = "fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634"
target = Path(__file__).resolve().parents[1] / "models" / name
target.parent.mkdir(exist_ok=True)

def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()

if target.exists():
    if digest(target) != expected:
        raise SystemExit("Existing model has a different checksum; move it aside before retrying.")
else:
    url = f"https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/{revision}/{name}"
    print("Downloading 3.35 GB from Google's public model repository...", flush=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=name + ".", suffix=".part", delete=False) as output:
        temporary = Path(output.name)
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                while chunk := response.read(8 * 1024 * 1024):
                    output.write(chunk)
            output.close()
            if digest(temporary) != expected:
                raise SystemExit("Download checksum mismatch; temporary file removed.")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
print(f"Verified: {target}")
