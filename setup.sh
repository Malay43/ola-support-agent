#!/usr/bin/env bash
# One-time setup. Python 3.10-3.13 required (tested on 3.12).
set -e
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# autogen is installed separately: autogen-core pins protobuf~=5.29 while crewai's OpenTelemetry stack needs protobuf>=6.33.5.
# pip prints a dependency-conflict WARNING here; it is expected. We then restore protobuf 6+, which both libraries work with (verified).
pip install "autogen-agentchat==0.7.5" "autogen-core==0.7.5" || true
pip install "protobuf>=6.33.5"
python - <<'PY'
import warnings; warnings.filterwarnings("ignore")
from autogen_agentchat.teams import RoundRobinGroupChat
from crewai.llms.base_llm import BaseLLM
import chromadb, fastapi
print("OK: crewai, autogen, chromadb, fastapi all import together")
PY
echo "Setup done. Next: source venv/bin/activate && python run_all.py && python make_readme.py"
