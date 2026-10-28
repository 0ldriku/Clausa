# macOS environment locks

Exact package lists of the three environments on Apple Silicon, made like those
in `container/locks/`.

`wheels-qwen3-asr.txt` lists the macOS 14 builds of MLX, mlx-metal and orjson.
The app needs them to run on macOS 14; pip would pick builds for the newest
macOS. To install them:

```bash
conda run -n qwen3-asr python -m pip download --no-deps --only-binary=:all: \
    --python-version 3.12 --implementation cp --abi cp312 --platform macosx_14_0_arm64 \
    -d wheels mlx==0.32.3 mlx-metal==0.32.3 orjson==3.13.0
conda run -n qwen3-asr python -m pip install --no-deps --force-reinstall wheels/*.whl
```

To rebuild an environment exactly:

```bash
conda create -n qwen3-asr --file conda-qwen3-asr.txt
conda run -n qwen3-asr python -m pip install --no-deps -r pip-qwen3-asr.txt
```

The English spaCy model `en_ud_L1L2e_combined_trf` is not on PyPI;
`runner/setup_envs.py` downloads it.
