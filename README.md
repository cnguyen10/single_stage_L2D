This is unofficial implementation of the following two papers:
1. Verma, Rajeev, and Eric Nalisnick. "Calibrated learning to defer with one-vs-all classifiers." In *International Conference on Machine Learning*, pp. 22184-22202. PMLR, 2022.
2. Verma, Rajeev, Daniel Barrejón, and Eric Nalisnick. "Learning to defer to multiple experts: Consistent surrogate losses, confidence calibration, and conformal ensembles." In *International Conference on Artificial Intelligence and Statistics*, pp. 11415-11434. PMLR, 2023.

The second paper extends the setting from single human expert in the first paper to multiple human experts. Both of the papers follow the single-stage without having a rejector (also known as gating). The reason for this implementation is due to the complexity of the original implementation made by the authors. In fact, the loss function can be calculated easily by augmenting the annotation vector without the need of appending or storing in a list.

## Packages
The implementation is mainly based on Jax and its eco-system as follows:
```bash
pip3 install -U "jax[cuda12]" --no-compile --no-cache-dir
pip3 install flax --no-compile --no-cache-dir
pip3 install optax --no-compile --no-cache-dir
pip3 install orbax-checkpoint --no-compile --no-cache-dir
pip3 install hydra-core --no-compile --no-cache-dir
pip3 install mlflow --no-compile --no-cache-dir
pip3 install tqdm --no-compile --no-cache-dir
pip3 install grain --no-compile --no-cache-dir
pip3 install albumentations --no-compile --no-cache-dir
pip3 install Pillow --no-compile --no-cache-dir
```
Note that the options `--no-compile` and `--no-cache-dir` are to reduce the size when creating a container in Docker or Apptainer.

## Data loading
The data loading is written to be more framework-agnostic through JSON files. In particular, each JSON file is a list of "dictionary" objects, each will have the following format:
```json
[
    {
        "file": "<path_to_one_image_file>",
        "label": "<true_label_in_int>"
    },
]
```

## Experiment tracking and management
The experiments are managed through `MLFlow`. Please see `mlflow_server.sh` for more details.