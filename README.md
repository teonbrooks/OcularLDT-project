OcularLDT-code
=====

A repo contains all the scripts involved with the analysis for the OcularLDT task, a project aimed to co-register of eye-movements with MEG data using a continuous lexical decision task [cf. [Hoedemaker and Gordon (2014)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4244277/)].


To start, please download the corresponding dataset on OpenNeuro [here](https://openneuro.org/datasets/ds002312).

Once you have the dataset, please set the path of the dataset in the `scripts/config.json`.
- `bids_root`
- `mri_root`: the path of the MRI
- `project_path`: the path of this repository

## Installing dependency

To install the dependencies, use the following commands. Note, with flag `--all-extras`, this will include dependencies such as `ipython`

```bash
uv venv
uv pip install -r pyproject.toml --all-extras
```
