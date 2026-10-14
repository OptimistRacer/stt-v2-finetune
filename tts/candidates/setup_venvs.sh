#!/bin/bash
# One isolated Python 3.11 environment per candidate, so no install can break the main
# Python 3.14 setup (installs have silently broken torch on this machine before).
# RTX 5090 (Blackwell) needs a CUDA 12.8 torch build; many packages pin older torch,
# so torch is installed first from the cu128 index and each package after it.
PY=/c/Users/Sistems-AI/AppData/Local/Programs/Python/Python311/python.exe
C=/d/Audion-Data/TTS/candidates
declare -A PKG=(
  [parler]="git+https://github.com/huggingface/parler-tts.git"
  [omnivoice]="omnivoice"
  [voxcpm]="voxcpm"
  [chatterbox]="chatterbox-tts"
)
for name in parler omnivoice voxcpm chatterbox; do
  echo "=== $name"
  $PY -m venv $C/venv_$name
  V=$C/venv_$name/Scripts/python.exe
  $V -m pip install -q --upgrade pip
  $V -m pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu128 || { echo "FAIL torch $name"; continue; }
  $V -m pip install -q ${PKG[$name]} || echo "FAIL pkg $name"
  # the package may have downgraded torch to a build without Blackwell support; check and repair
  $V -c "import torch; ok=torch.cuda.is_available() and 'sm_120' in torch.cuda.get_arch_list(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'blackwell', ok)" 2>&1 | tail -1
  if ! $V -c "import torch,sys; sys.exit(0 if 'sm_120' in torch.cuda.get_arch_list() else 1)" 2>/dev/null; then
    echo "repairing torch for $name"
    $V -m pip install -q --force-reinstall --no-deps torch torchaudio --index-url https://download.pytorch.org/whl/cu128
    $V -c "import torch; print('torch', torch.__version__, 'blackwell', 'sm_120' in torch.cuda.get_arch_list())"
  fi
done
echo "=== ALL VENVS DONE"
