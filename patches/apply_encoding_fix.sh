#!/usr/bin/env bash
# Replace encoding_dsv41.py with upstream SGLang commit a038a0dca (#44751: tool-result image order and current-turn
# thinking), merged after the 2026-10-16 nightly. Only if the installed file is exactly the commit's parent.
# Usage: bash patches/apply_encoding_fix.sh /sgl-workspace/sglang
set -euo pipefail
SRC="${1:?path to the sglang source tree}"
F="$SRC/python/sglang/srt/entrypoints/openai/encoding_dsv41.py"
PARENT_BLOB=c04d8059d81b5b943c017728817b1e880e22adbb   # git blob of the file at a038a0dca~1 (= nightly a93f16f0)
have=$(git hash-object "$F")
if [ "$have" != "$PARENT_BLOB" ]; then echo "installed encoding_dsv41.py is not the expected base ($have); not patching"; exit 1; fi
cp "$F" "$F.orig"
curl -fsSL https://raw.githubusercontent.com/sgl-project/sglang/a038a0dca/python/sglang/srt/entrypoints/openai/encoding_dsv41.py -o "$F"
python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$F"
echo "applied a038a0dca encoding_dsv41.py (original kept as $F.orig)"
