# Windows setup (PowerShell). Python 3.10-3.13 required (tested on 3.12).
# If activation is blocked, first run:  Set-ExecutionPolicy -Scope Process Bypass
$ErrorActionPreference = "Stop"
python -m venv venv
& .\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# autogen is installed separately: autogen-core pins protobuf~=5.29 while crewai needs protobuf>=6.33.5.
# pip prints a dependency-conflict WARNING here; it is expected. We restore protobuf 6+ afterwards (verified to work).
try { pip install "autogen-agentchat==0.7.5" "autogen-core==0.7.5" } catch { }
pip install "protobuf>=6.33.5"
python -c "import warnings; warnings.filterwarnings('ignore'); from autogen_agentchat.teams import RoundRobinGroupChat; from crewai.llms.base_llm import BaseLLM; import chromadb, fastapi; print('OK: crewai, autogen, chromadb, fastapi all import together')"
Write-Host "Setup done. Next:  `$env:PYTHONUTF8='1'; python run_all.py; python make_readme.py"
