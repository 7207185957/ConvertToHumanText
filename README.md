# ConvertToHumanText

`humanizer-agent` is a lightweight Python agent that:

1. Converts AI-generated text into more human-like writing.
2. Verifies the converted output against reference examples.
3. Supports reference examples from:
   - text files (`.txt`, `.md`)
   - JSON arrays of strings (`.json`)
   - images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`) via OCR

> The verification threshold defaults to **99.9** (configurable).  
> In practice, exact real-world accuracy guarantees depend on your reference data quality.

## Quick Start

### Web UI (split-screen, HIX-style flow)

Install dependencies:

```bash
pip install -e .
```

Start UI:

```bash
python -m humanizer_agent.webapp
```

Then open: `http://localhost:8000`

- Left side: paste AI-generated text.
- Right side: see humanized text output.
- Optional: add reference text or upload a reference file/image for verification.

### 1) Run with plain input text

```bash
python -m humanizer_agent.cli \
  --text "Moreover, it is important that we do not utilize overly complex terminology." \
  --json
```

### 2) Run with references from image (OCR)

Install optional OCR dependencies first:

```bash
pip install .[ocr]
```

Then run:

```bash
python -m humanizer_agent.cli \
  --input-file ./ai_text.txt \
  --reference-source ./reference_examples.png \
  --min-score 99.9 \
  --report-file ./verification_report.json
```

### 3) Run with inline reference examples

```bash
python -m humanizer_agent.cli \
  --text "Therefore, this methodology is sufficient for the majority of users." \
  --references-text $'I usually keep wording simple.\nThis is good enough for most people.' \
  --min-score 85
```

## CLI Options

- `--text`: AI-generated input text.
- `--input-file`: path to input file (mutually exclusive with `--text`).
- `--reference-source`: path to text/json/image references.
- `--references-text`: inline newline-separated references.
- `--min-score`: minimum verification score required (default `99.9`).
- `--max-iterations`: rewrite attempts for style matching (default `4`).
- `--output-file`: save only converted text.
- `--report-file`: save full JSON report.
- `--json`: print full JSON report to stdout.

## Python API

```python
from humanizer_agent import AITextHumanizationAgent

agent = AITextHumanizationAgent(min_verification_score=99.9)
result = agent.convert(
    ai_generated_text="Furthermore, it is important to utilize precise terminology.",
    reference_source="reference_examples.png",  # or .txt/.json
)

print(result.humanized_text)
print(result.to_dict())
```

## Run Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```