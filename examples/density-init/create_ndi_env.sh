#!/usr/bin/env bash
# Create an isolated environment for the neural_paw_dft initializer.
#
# neural_paw_dft pins torch/mace-torch==0.3.16/e3nn==0.4.4/pykeops, which
# conflicts with interfaceforge[mace-roi] (mace-torch>=0.3.17), so it lives in
# its own interpreter; InterfaceForge calls it via --ndi-python.
#
# Usage: ./create_ndi_env.sh ~/envs/ndi [cu124|cpu]
set -euo pipefail
PREFIX=${1:?usage: create_ndi_env.sh PREFIX [cu124|cpu]}
FLAVOUR=${2:-cu124}
python3 -m venv "$PREFIX"
if [ "$FLAVOUR" = "cpu" ]; then
    "$PREFIX/bin/pip" install "torch>=2.4.1" --index-url https://download.pytorch.org/whl/cpu
else
    "$PREFIX/bin/pip" install "torch>=2.4.1" --index-url "https://download.pytorch.org/whl/$FLAVOUR"
fi
# Pin the commit the InterfaceForge adapter was written against.  The packaged
# `neural_paw_dft.pipeline` API (Pipeline.predict + assemble.build_chgcar) exists
# only on `main`; the paper tag `v1-paper` predates it and will NOT work.
NDI_COMMIT=${NDI_COMMIT:-bde513bf8eb793c1ff339145e78384540aaebe6d}
"$PREFIX/bin/pip" install "neural_paw_dft @ git+https://github.com/aerte/neural_paw_dft@$NDI_COMMIT"
cat <<MSG

Created $PREFIX.  Next (on a node with network access, e.g. the login node):
  iface vasp density-init-probe --ndi-python $PREFIX/bin/python --prefetch --weights-dir /shared/ndi_weights
pykeops JIT-compiles on first use and needs a C++ compiler on the compute node
(module load gcc); run one inference there before a production campaign.
MSG
