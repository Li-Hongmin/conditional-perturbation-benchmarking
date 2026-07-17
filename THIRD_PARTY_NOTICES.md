# Third-party notices

This repository does not vendor or redistribute the packages below. They are
installed separately from package indexes and remain governed by their
respective licenses. The project license does not supersede those licenses.

| Direct dependency | Role | License | Official license source |
| --- | --- | --- | --- |
| Hatchling | build backend | MIT | https://github.com/pypa/hatch/blob/master/LICENSE.txt |
| Matplotlib 3.11.0 | plotting | Matplotlib License (PSF-style) | https://github.com/matplotlib/matplotlib/blob/v3.11.0/LICENSE/LICENSE |
| NumPy 2.4.6 | numerical arrays | BSD-3-Clause | https://github.com/numpy/numpy/blob/v2.4.6/LICENSE.txt |
| pandas 2.3.3 | tabular data | BSD-3-Clause | https://github.com/pandas-dev/pandas/blob/v2.3.3/LICENSE |
| Pillow 12.3.0 | image support | MIT-CMU | https://github.com/python-pillow/Pillow/blob/12.3.0/LICENSE |
| pytest 9.1.1 | tests | MIT | https://github.com/pytest-dev/pytest/blob/9.1.1/LICENSE |
| SciPy 1.17.1 | statistical functions | BSD-3-Clause | https://github.com/scipy/scipy/blob/v1.17.1/LICENSE.txt |

Transitive dependencies are recorded in `uv.lock`; this table lists the
project's direct declared dependencies.

## Upstream research software and data

- Norman combinatorial Perturb-seq: GEO `GSE133344`; Norman et al., Science
  (2019), DOI `10.1126/science.aax4438`. The expression matrix is not
  redistributed.
- PerturBench: `https://github.com/altoslabs/perturbench`, commit
  `4825e392294768da4b35561a76502c7006d6453e`, BSD-3-Clause with
  model-specific notices. Its implementations and configuration files are not
  redistributed by this public replay; only their names and hashes are cited.
