# Dev setup quickstart

This is the minimal path to run the agent locally.

## Prerequisites

- Python 3.10+
- Robot Framework
- A UI automation backend:
  - **Web**: Playwright + a browser driver
  - **Mobile**: Appium + real device/emulator
- Access to an LLM provider (OpenAI/Azure/local), depending on config.

## Install

```bash
git clone <REPO_URL>
cd <REPO_NAME>

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Configure environment

Create a local env file or RF variable file (example name):

- `config/local.yaml`
- or `variables.py`
- or `variables.robot`

Set at least:

- LLM provider + key
- target platform (web / android / ios)
- device/browser settings
- artifact output path

Example (pseudo):

```yaml
llm:
  provider: openai
  model: gpt-4.1-mini
  api_key: "<YOUR_KEY>"

runtime:
  platform: android
  artifacts_dir: "./artifacts"
```

## Run a sample suite

```bash
robot -L TRACE tests/agent_smoke.robot
```

## Outputs

After a run you should get:

- Robot log + report
- screenshots per step
- agent reasoning trace (if enabled)

Artifacts location:

```
./artifacts/<run_id>/...
```

## Common issues

- Appium/Selenium not running
- Wrong device/browser capabilities
- Missing API key / model name
- Permission issues for screenshots

