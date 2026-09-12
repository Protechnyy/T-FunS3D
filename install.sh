#!/usr/bin/env bash
set -eo pipefail

# Run from the repository root.
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda run -n T-Search3D python --version >/dev/null 2>&1; then
    conda create -y -n T-Search3D python=3.10
fi
conda activate T-Search3D

conda install -y -c nvidia/label/cuda-12.1.1 cuda-toolkit=12.1.1
conda install -y gcc_linux-64=11 gxx_linux-64=11
conda install -y -c anaconda openblas-devel
export CUDA_HOME="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"

python -m pip install setuptools==59.8.0 wheel==0.45.1
python -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pyyaml_constraint_file=$(mktemp)
printf 'Cython<3\n' > "$pyyaml_constraint_file"
PIP_CONSTRAINT="$pyyaml_constraint_file" python -m pip install PyYAML==5.4.1
rm -f "$pyyaml_constraint_file"
python -m pip install numpy==1.24.4 pytorch-lightning==1.9.5 torchmetrics==0.11.4 hydra-core==1.3.2 omegaconf==2.3.0 antlr4-python3-runtime==4.9.3
python -m pip install PyYAML==5.4.1 transformers==4.51.3 numba==0.60.0 ninja==1.10.2.3 fire imageio
python -m pip install tqdm wandb python-dotenv==0.21.0 pyviz3d==0.2.32 scipy==1.9.3 plyfile==0.7.4
python -m pip install scikit-learn==1.2.0 trimesh==3.17.1 loguru==0.6.0 albumentations==1.3.0 volumentations==0.1.8 black==21.4b2
python -m pip install pynvml==11.4.1 gpustat==1.0.0 tabulate==0.9.0 pytest==7.2.0 tensorboardx==2.6.4 yapf==0.32.0
python -m pip install termcolor==2.1.1 addict==2.4.0 blessed==1.19.1 gorilla-core==0.2.7.8 matplotlib==3.7.2 'Cython<3'
python -m pip install pycocotools==2.0.6 h5py==3.7.0 transforms3d==0.4.1 open3d==0.16.0 fvcore cloudpickle
python -m pip install Pillow ftfy regex spacy==3.7.2 einops shapely
python -m pip install timm yacs tensorboard prettytable pymongo nltk
python -m pip install inflect appdirs mypy-extensions pathspec toml sentencepiece
python -m pip install accelerate==0.34.2 iopath==0.1.9 protobuf==6.33.5

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@ff53992b1985b63bd3262b5a36167098e3dada02' --no-cache-dir --no-deps --no-build-isolation
# CUDA 12 no longer includes these Thrust declarations transitively.
NVCC_PREPEND_FLAGS="--pre-include=thrust/execution_policy.h,thrust/remove.h,thrust/unique.h,thrust/sort.h ${NVCC_PREPEND_FLAGS:-}" \
python -m pip install 'git+https://github.com/NVIDIA/MinkowskiEngine@02fc608bea4c0549b0a7b00ca1bf15dee4a0b228' --no-deps --no-build-isolation \
    --config-settings="--global-option=--blas_include_dirs=${CONDA_PREFIX}/include" \
    --config-settings="--global-option=--blas=openblas"
python -m pip install torch-scatter==2.1.2+pt21cu121 --no-deps --only-binary=:all: -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
(
    cd third-party/openmask3d/openmask3d/class_agnostic_mask_computation/third_party/pointnet2
    python -m pip install . --no-deps --no-build-isolation
)
python -m pip install 'git+https://github.com/openai/CLIP.git@a9b1bf5920416aaeaec965c25dd9e8f98c864f16' --no-deps --no-build-isolation
python -m pip install 'git+https://github.com/facebookresearch/segment-anything.git@6fdee8f2727f4506cfbbe553e23b895e27956588' --no-deps --no-build-isolation
# Editable installs expose the subpackages in these source trees.
python -m pip install -e third-party/openmask3d -e . --no-deps

clone_pinned() {
    local url="$1" dest="$2" revision="$3"
    if [[ ! -d "$dest/.git" ]]; then
        git clone "$url" "$dest"
        git -C "$dest" checkout "$revision"
    fi
    [[ "$(git -C "$dest" rev-parse HEAD)" == "$revision" ]] || {
        echo "Unexpected revision in $dest; expected $revision" >&2
        return 1
    }
}
clone_pinned https://github.com/Karbo123/segmentator.git third-party/segmentator 4c6126551685166c6c300551e9ad63db988928c4
clone_pinned https://github.com/microsoft/GLIP.git third-party/GLIP 9dda9558c1ef59bb6cdc8e896e2bcab775a68ff0
python - <<'PY_SETUP'
"""Apply idempotent compatibility fixes to the pinned build dependencies."""
from pathlib import Path
import re

def write_changed(path, text):
    if path.read_text() != text:
        path.write_text(text)


root = Path.cwd()
segment = root / 'third-party/segmentator/csrc/CMakeLists.txt'
if segment.exists():
    text = segment.read_text().replace('set(CMAKE_CXX_STANDARD 14)', 'set(CMAKE_CXX_STANDARD 17)')
    write_changed(segment, text)

for path in (root / 'third-party/GLIP/maskrcnn_benchmark/csrc').rglob('*'):
    if path.suffix not in {'.cu', '.cpp', '.h', '.cuh'}:
        continue
    text = path.read_text()
    replacements = {
        '#include <THC/THC.h>': '#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAException.h>\n#include <c10/cuda/CUDACachingAllocator.h>',
        '#include <THC/THCAtomics.cuh>': '#include <ATen/cuda/Atomic.cuh>',
        '#include <THC/THCDeviceUtils.cuh>': '#include <ATen/ceil_div.h>',
        'THCCeilDiv': 'at::ceil_div',
        'THCudaCheck': 'C10_CUDA_CHECK',
        'THCudaMalloc(state, ': 'c10::cuda::CUDACachingAllocator::raw_alloc(',
        'THCudaFree(state, ': 'c10::cuda::CUDACachingAllocator::raw_delete(',
        'AT_CHECK(': 'TORCH_CHECK(',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r'^.*THCState \*state =.*\n', '', text, flags=re.MULTILINE)
    write_changed(path, text)
setup = root / 'third-party/GLIP/setup.py'
# GLIP still references the removed torch._six Python-version helpers.
for path in (root / 'third-party/GLIP/maskrcnn_benchmark').rglob('*.py'):
    text = path.read_text()
    text = text.replace('torch._six.PY37', 'True').replace('torch._six.PY3', 'True')
    text = text.replace('import torch._six\n', '')
    write_changed(path, text)

# Data is provisioned at install time; importing the predictor need not fetch it.
predictor = root / 'third-party/GLIP/maskrcnn_benchmark/engine/predictor_glip.py'
if predictor.exists():
    text = predictor.read_text()
    for name in ['punkt', 'averaged_perceptron_tagger']:
        text = text.replace(f"nltk.download('{name}')", '# NLTK data is provisioned by install.sh.')
    write_changed(predictor, text)

# Include Python subdirectories that lack __init__.py in GLIP's wheel.
if setup.exists():
    text = setup.read_text().replace('from setuptools import find_packages', 'from setuptools import find_namespace_packages')
    text = text.replace('find_packages(exclude=("configs", "tests",))', 'find_namespace_packages(include=["maskrcnn_benchmark*"])')
    write_changed(setup, text)

model_zoo = root / 'third-party/GLIP/maskrcnn_benchmark/utils/model_zoo.py'
if model_zoo.exists():
    text = model_zoo.read_text().replace('from torch.hub import _download_url_to_file', 'from torch.hub import download_url_to_file as _download_url_to_file')
    write_changed(model_zoo, text)
PY_SETUP
python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng
cmake -S third-party/segmentator/csrc -B third-party/segmentator/csrc/build \
    -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
    -DPYTHON_INCLUDE_DIR="$CONDA_PREFIX/include/python3.10" \
    -DPYTHON_LIBRARY="$CONDA_PREFIX/lib/libpython3.10.so" \
    -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX/lib/python3.10/site-packages"
cmake --build third-party/segmentator/csrc/build
cmake --install third-party/segmentator/csrc/build
# Keep third-party/segmentator: the installed module points to this directory.
python -m pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl' --no-deps
python -m pip install ./third-party/GLIP --no-deps --no-build-isolation
python -m pip check
