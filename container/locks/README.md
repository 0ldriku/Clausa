# Environment locks

Exact package lists of the Linux image that `container/Containerfile` builds.
`conda-<env>.txt` comes from `micromamba env export --explicit`; `pip-<env>.txt`
lists only what pip installed on top.

`./clausa setup` pins only top-level packages, so a new install can get newer
dependencies. To rebuild an environment exactly:

```bash
micromamba create -n mfa --file conda-mfa.txt
micromamba run -n mfa python -m pip install --no-deps -r pip-mfa.txt
```

The CUDA wheels of PyTorch need `--extra-index-url https://download.pytorch.org/whl/cu128`.
The English spaCy model `en_ud_L1L2e_combined_trf` is not on PyPI;
`runner/setup_envs.py` downloads it.
