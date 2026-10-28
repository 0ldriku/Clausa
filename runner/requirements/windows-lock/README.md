# Windows environment locks

Exact package lists of the three environments on Windows, made like those in
`../macos-lock/`. The pinned packages and MFA's stack have the same versions
as in the Linux container. PostgreSQL is 16 instead of 18; MFA does not use it.
`qwen3-asr` also has the app window's packages, which these lists leave out:
`pywebview==6.2.1`, `bottle==0.13.4`, `clr_loader==0.2.10`,
`proxy_tools==0.1.0` and `pythonnet==3.0.5`.

To rebuild an environment exactly:

```bat
micromamba create -n qwen3-asr --file conda-qwen3-asr.txt
micromamba run -n qwen3-asr python -m pip install --no-deps -r pip-qwen3-asr.txt ^
    --extra-index-url https://download.pytorch.org/whl/cu128
```

For `ja-electra310`, use `--extra-index-url https://download.pytorch.org/whl/cpu`.
The English spaCy model `en_ud_L1L2e_combined_trf` is not on PyPI;
`runner/setup_envs.py` downloads it.
