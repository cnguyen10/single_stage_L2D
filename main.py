import os
from pathlib import Path
from functools import partial

from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

import jax
import jax.numpy as jnp

import flax.nnx as nnx
from flax.nnx import metrics
from flax.traverse_util import flatten_dict

import orbax.checkpoint as ocp

import optax

import grain.python as grain

import mlflow

from DataSource import ImageDataSource
from utils import (
    init_tx,
    initialize_dataloader
)


@partial(jax.jit, static_argnames=('num_classes',))
def augment_labels(y: jax.Array, t: jax.Array, num_classes: int) -> jax.Array:
    """augment the labels for the unified gating + classifier model

    Args:
        y: ground truth labels (batch,)
        t: expert's annotations (missing is denoted as -1) (batch, num_experts)
        num_classes:

    Return:
        y_augmented:
    """
    y_one_hot = jax.nn.one_hot(
        x=y,
        num_classes=num_classes
    )  # (batch, num_classes)

    # binary flag of expert's predictions
    y_orthogonal = (t == y[:, None]) * 1  # (batch, num_experts)

    y_augmented = jnp.concatenate(
        arrays=(y_one_hot, y_orthogonal),
        axis=-1
    )  # (batch, num_classes + num_experts)

    return y_augmented


@partial(nnx.jit, static_argnames=('num_classes', 'dirichlet_concentration'))
def calculate_loss_dirichlet_prior(
    model: nnx.Module,
    x: jax.Array,
    num_classes: int,
    dirichlet_concentration: list[float]
) -> jax.Array:
    """calculate the prior imposed on the prediction of the classifier.
    The purpose of the prior loss is to control the coverage by changing
    the Dirichlet concentration

    Args:
        model: the 'unified' model whose output consists of the prediction of
            the classifier and the correction of human experts.
        x: the input samples
        num_classes: the number of classes in the classification task
        dirichlet_concentration: a (num_experts + 1)-dimensional positive
            vector

    Returns:
        loss_prior:
    """
    logits = model(x)
    log_softmax = jax.nn.log_softmax(x=logits, axis=-1)
    log_softmax_clf = jax.nn.logsumexp(a=log_softmax[:, :num_classes], axis=-1)
    log_logits_gating = jnp.concatenate(
        arrays=(log_softmax[:, num_classes:], log_softmax_clf[:, None]),
        axis=-1
    )

    loss_prior = - jnp.sum(
        a=(jnp.array(object=dirichlet_concentration) - 1) * log_logits_gating,
        axis=-1
    )
    loss_prior = jnp.mean(a=loss_prior, axis=0)

    return loss_prior


@nnx.jit
def softmax_loss_fn(
    model: nnx.Module,
    x: jax.Array,
    y_augmented: jax.Array
) -> jax.Array:
    """calculate the loss of the single-staged L2D with softmax
    """
    logits = model(x)

    loss = optax.losses.softmax_cross_entropy(
        logits=logits,
        labels=y_augmented
    )
    loss = jnp.mean(a=loss, axis=0)

    return loss


@nnx.jit
def one_vs_all(
    model: nnx.Module,
    x: jax.Array,
    y_augmented: jax.Array
) -> jax.Array:
    """
    """
    logits = model(x)

    loss = optax.losses.sigmoid_binary_cross_entropy(
        logits=logits,
        labels=y_augmented
    )
    loss = jnp.sum(a=loss, axis=-1)
    loss = jnp.mean(a=loss, axis=0)

    return loss


@partial(
    nnx.jit,
    static_argnames=(
        'which_loss',
        'num_classes',
        'dirichlet_concentration',
        'dataset_length'
    )
)
def loss_fn(
    model: nnx.Module,
    x: jax.Array,
    y_augmented: jax.Array,
    which_loss: str,
    num_classes: int,
    dirichlet_concentration: list[float],
    dataset_length: int
) -> jax.Array:
    """a wrapper to calculate the loss

    Args:
        model: the 'unified" model
        x: input samples
        y_augmented: the first num_classes elements corresponds to the ground
            truth label to train the classifier, while the remainings
            correspond to the correction of each human expert
        which_loss: either 'softmax_loss_fn' or 'one_vs_all'
        num_classes:
        dirichlet_concentration:
        dataset_length

    Returns:
        loss: the total loss including the prior as well
    """
    match which_loss:
        case 'softmax':
            loss_defer = softmax_loss_fn(
                model=model,
                x=x,
                y_augmented=y_augmented
            )
            loss_prior = calculate_loss_dirichlet_prior(
                model=model,
                x=x,
                num_classes=num_classes,
                dirichlet_concentration=dirichlet_concentration
            )
            loss = loss_defer + (len(x) / dataset_length) * loss_prior

        case 'one_vs_all':
            loss = one_vs_all(model=model, x=x, y_augmented=y_augmented)

        case _:
            raise ValueError(' '.join((
                'The loss function must be either',
                '\'softmax\' or \'one_vs_all\'.'
                f'Found {which_loss}'
            )))

    return loss


@partial(
    nnx.jit,
    static_argnames=(
        'which_loss',
        'num_classes',
        'dirichlet_concentration',
        'dataset_length'
    )
)
def train_step(
    x: jax.Array,
    y_augmented: jax.Array,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    which_loss: str,
    num_classes: int,
    dirichlet_concentration: list[float],
    dataset_length: int
) -> tuple[nnx.Module, nnx.Optimizer, jax.Array]:
    """
    """
    grad_value_fn = nnx.value_and_grad(f=loss_fn, argnums=0)
    loss, grads = grad_value_fn(
        model,
        x,
        y_augmented,
        which_loss,
        num_classes,
        dirichlet_concentration,
        dataset_length
    )

    optimizer.update(model=model, grads=grads)

    return (model, optimizer, loss)


def train(
    dataloader: grain.DatasetIterator,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    cfg: DictConfig
) -> tuple[nnx.Module, nnx.Optimizer, jax.Array]:
    """
    """
    # metric to track the training loss
    loss_accum = metrics.Average()

    model.train()

    for _ in tqdm(
        iterable=range(cfg.dataset.length.train // cfg.training.batch_size),
        desc='epoch',
        ncols=80,
        leave=False,
        position=2,
        colour='blue',
        disable=not cfg.data_loading.progress_bar
    ):
        samples = next(dataloader)

        x = jnp.asarray(
            a=samples['image'],
            dtype=eval(cfg.jax.dtype)
        )  # input samples
        y = jnp.asarray(
            a=samples['ground_truth'],
            dtype=jnp.int32
        )  # true int labels  (batch,)
        t = jnp.asarray(
            a=samples['label'],
            dtype=jnp.int32
        )  # annotated int labels (batch, num_experts)

        # augmented labels
        y_augmented = augment_labels(
            y=y,
            t=t,
            num_classes=cfg.dataset.num_classes
        )

        model, optimizer, loss = train_step(
            x=x,
            y_augmented=y_augmented,
            model=model,
            optimizer=optimizer,
            which_loss=cfg.training.loss_fn,
            num_classes=cfg.dataset.num_classes,
            dirichlet_concentration=cfg.hparams.dirichlet_concentration,
            dataset_length=cfg.dataset.length.train
        )

        if jnp.isnan(loss):
            raise ValueError('Training loss is NaN.')

        # tracking
        loss_accum.update(values=loss)

    return model, optimizer, loss_accum.compute()


def evaluate(
    dataloader: grain.DataLoader,
    model: nnx.Module,
    cfg: DictConfig
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    """
    accuracy_accum = metrics.Accuracy()
    coverage = metrics.Average()
    clf_accuracy_accum = metrics.Accuracy()

    model.eval()

    for samples in tqdm(
        iterable=dataloader,
        total=cfg.dataset.length.test//cfg.training.batch_size + 1,
        desc='evaluate',
        ncols=80,
        leave=False,
        position=2,
        colour='blue',
        disable=not cfg.data_loading.progress_bar
    ):
        x = jnp.asarray(
            a=samples['image'],
            dtype=jnp.float32
        )  # input samples
        y = jnp.asarray(
            a=samples['ground_truth'],
            dtype=jnp.int32
        )  # true labels (batch,)

        # annotated labels (batch, num_experts)
        t = jnp.asarray(a=samples['label'], dtype=jnp.int32)

        logits = model(x)  # (batch, num_classes + num_experts)

        # classifier predictions
        clf_predictions = jnp.argmax(
            a=logits[:, :cfg.dataset.num_classes],
            axis=-1
        )  # (batch,)
        clf_accuracy_accum.update(
            logits=logits[:, :cfg.dataset.num_classes],
            labels=y
        )

        labels_concatenated = jnp.concatenate(
            arrays=(t, clf_predictions[:, None]),
            axis=-1
        )  # (batch, num_experts + 1)

        logits_max_id = jnp.argmax(a=logits, axis=-1)  # (batch,)
        logits_max_id = logits_max_id - cfg.dataset.num_classes

        # which samples are predicted by classifier
        samples_predicted_by_clf = (logits_max_id < 0) * 1  # (batch,)

        # which samples are deferred to which experts  # (batch,)
        sample_expert_id = logits_max_id * (1 - samples_predicted_by_clf)

        selected_expert_ids = sample_expert_id\
            + samples_predicted_by_clf * len(cfg.dataset.test_files)

        coverage.update(values=samples_predicted_by_clf)

        # system's predictions
        y_predicted = labels_concatenated[
            jnp.arange(y.shape[0]),
            selected_expert_ids
        ]

        accuracy_accum.update(
            logits=jax.nn.one_hot(
                x=y_predicted,
                num_classes=cfg.dataset.num_classes
            ),
            labels=y
        )

    return (
        accuracy_accum.compute(),
        coverage.compute(),
        clf_accuracy_accum.compute()
    )


@hydra.main(version_base=None, config_path='conf', config_name='conf')
def main(cfg: DictConfig) -> None:
    """
    """
    jax.config.update('jax_disable_jit', cfg.jax.disable_jit)
    jax.config.update('jax_platforms', cfg.jax.platform)

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(cfg.jax.mem)

    # region DATASETS
    source_train = ImageDataSource(
        annotation_files=cfg.dataset.train_files,
        ground_truth_file=cfg.dataset.train_ground_truth_file,
        root=cfg.dataset.root,
        num_samples=cfg.training.num_samples,
        seed=cfg.training.seed
    )
    source_test = ImageDataSource(
        annotation_files=cfg.dataset.test_files,
        ground_truth_file=cfg.dataset.test_ground_truth_file,
        root=cfg.dataset.root
    )

    OmegaConf.set_struct(conf=cfg, value=True)
    OmegaConf.update(
        cfg=cfg,
        key='dataset.length.train',
        value=len(source_train),
        force_add=True
    )
    OmegaConf.update(
        cfg=cfg,
        key='dataset.length.test',
        value=len(source_test),
        force_add=True
    )
    # endregion

    # region MODELS
    model = hydra.utils.instantiate(config=cfg.model)(
        num_classes=cfg.dataset.num_classes + len(cfg.dataset.train_files),
        rngs=nnx.Rngs(jax.random.PRNGKey(seed=cfg.training.seed)),
        dropout_rate=cfg.training.dropout_rate,
        dtype=eval(cfg.jax.dtype)
    )

    optimizer = nnx.Optimizer(
        model=model,
        tx=init_tx(
            dataset_length=len(source_train),
            lr=cfg.training.lr,
            batch_size=cfg.training.batch_size,
            num_epochs=cfg.training.num_epochs,
            weight_decay=cfg.training.weight_decay,
            momentum=cfg.training.momentum,
            clipped_norm=cfg.training.clipped_norm,
            key=cfg.training.seed
        ),
        wrt=nnx.Param
    )
    # endregion

    # options to store models
    ckpt_options = ocp.CheckpointManagerOptions(
        save_interval_steps=100,
        max_to_keep=1,
        step_format_fixed_length=3,
        enable_async_checkpointing=True
    )

    # region Mlflow
    mlflow.set_tracking_uri(uri=cfg.experiment.tracking_uri)
    mlflow.set_experiment(experiment_name=cfg.experiment.name)
    mlflow.disable_system_metrics_logging()

    # create a directory for storage (if not existed)
    if not os.path.exists(path=cfg.experiment.logdir):
        Path(cfg.experiment.logdir).mkdir(parents=True, exist_ok=True)
    # endregion

    # enable mlflow tracking
    with mlflow.start_run(
            run_id=cfg.experiment.run_id,
            log_system_metrics=False) as mlflow_run, \
        ocp.CheckpointManager(
            directory=os.path.join(
                os.getcwd(),
                cfg.experiment.logdir,
                cfg.experiment.name,
                mlflow_run.info.run_id
            ),
            options=ckpt_options) as ckpt_mngr:
        if cfg.experiment.run_id is None:
            # log hyper-parameters
            mlflow.log_params(
                params=flatten_dict(
                    xs=OmegaConf.to_container(cfg=cfg),
                    sep='.')
            )

            # log source code
            mlflow.log_artifact(
                local_path=os.path.abspath(path=__file__),
                artifact_path='source_code'
            )

            start_epoch_id = 0
        else:
            start_epoch_id = ckpt_mngr.latest_step()

            checkpoint = ckpt_mngr.restore(
                step=start_epoch_id,
                args=ocp.args.StandardRestore(item=nnx.state(model))
            )

            nnx.update(model, checkpoint)

            del checkpoint

        # create iterative datasets as data loaders
        dataloader_train = initialize_dataloader(
            data_source=source_train,
            num_epochs=cfg.training.num_epochs - start_epoch_id + 1,
            shuffle=True,
            seed=cfg.training.seed,
            batch_size=cfg.training.batch_size,
            resize=cfg.data_augmentation.resize,
            padding_px=cfg.data_augmentation.padding_px,
            crop_size=cfg.data_augmentation.crop_size,
            mean=cfg.data_augmentation.mean,
            std=cfg.data_augmentation.std,
            p_flip=cfg.data_augmentation.prob_random_flip,
            num_workers=cfg.data_loading.num_workers,
            num_threads=cfg.data_loading.num_threads,
            prefetch_size=cfg.data_loading.prefetch_size
        )
        dataloader_train = iter(dataloader_train)

        dataloader_test = initialize_dataloader(
            data_source=source_test,
            num_epochs=1,
            shuffle=False,
            seed=0,
            batch_size=cfg.training.batch_size,
            resize=cfg.data_augmentation.crop_size,
            padding_px=None,
            crop_size=None,
            mean=cfg.data_augmentation.mean,
            std=cfg.data_augmentation.std,
            p_flip=None,
            is_color_img=True,
            num_workers=cfg.data_loading.num_workers,
            num_threads=cfg.data_loading.num_threads,
            prefetch_size=cfg.data_loading.prefetch_size
        )

        for epoch_id in tqdm(
            iterable=range(start_epoch_id, cfg.training.num_epochs, 1),
            desc='progress',
            ncols=80,
            leave=True,
            position=1,
            colour='green',
            disable=not cfg.data_loading.progress_bar
        ):
            model, optimizer, loss = train(
                dataloader=dataloader_train,
                model=model,
                optimizer=optimizer,
                cfg=cfg
            )

            # wait for the asynchronous saving
            ckpt_mngr.wait_until_finished()

            accuracy, coverage, clf_accuracy = evaluate(
                dataloader=dataloader_test,
                model=model,
                cfg=cfg
            )

            logged_metrics = dict(
                loss=loss,
                accuracy=accuracy,
                coverage=coverage,
                clf_accuracy=clf_accuracy
            )
            mlflow.log_metrics(
                metrics=logged_metrics,
                step=epoch_id + 1,
                synchronous=False
            )

            # save checkpoint
            ckpt_mngr.save(
                step=epoch_id + 1,
                args=ocp.args.StandardSave(nnx.state(model))
            )
    return None


if __name__ == '__main__':
    # cache jax-compiled file if the compiling takes more than 2 minutes
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 120)

    main()
