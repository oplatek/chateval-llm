#!/usr/bin/env python3
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader
from chateval.datasets import turn_metric_mapping, load_dataset as load_chateval_dataset
import argparse
import logging
import torch
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from chateval.results import (
    evaluate_metric,
    results_to_csv,
    results_to_wandb,
    exp_dirname,
)
from chateval.histogram import plot_histogram
from chateval.scatter_plot import plot_scatter
from nn_model import RegressionModule
from embed_dataset import EmbDataset


def predict(trainer: pl.Trainer, model: RegressionModule, dataset: EmbDataset, args: argparse.Namespace, outdir: Path, name: str):
    """The predict function is used in the training loop but can be used indepently if you have a checkpoint"""
    predictions = trainer.predict(
        model=model,
        dataloaders=DataLoader(
            dataset,
            args.batch_size,
            num_workers=args.dataloader_workers,
            shuffle=False,
            drop_last=False,
        ),
    )
    if len(predictions) == 0:
        logging.error(f"Predicting {name=} preduced 0 predictions")
        return

    has_gold_quality = f"annotations.{args.quality}" in predictions[0]
    quality = torch.cat([d[args.quality] for d in predictions]).cpu().numpy()
    if not has_gold_quality:
        logging.warning(f"The {name=} has no gold annotations for {args.quality}")
        # we produce -1 for 
        gold_quality = -1 * np.ones(quality.shape)
    else:
        gold_quality = torch.cat([d[f"annotations.{args.quality}"] for d in predictions]).cpu().numpy()

    dialogue_ids = [d["dialogue_id"] for d in predictions]
    dialogue_ids = [item for sublist in dialogue_ids for item in sublist]
    outpath = outdir / f"{name}_{args.quality}.csv"
    print(f"{outpath=}", flush=True)

    with open(outpath, "wt") as w:
        w.write(f"UID,{args.quality}\n")
        for (
            did,
            p,
            g,
        ) in zip(dialogue_ids, quality.tolist(), gold_quality.tolist()):
            w.write(f"{did},{p},{g}\n")

    if has_gold_quality:
        plot_scatter(gold_quality, quality, args.quality, f"{args.quality} on {name}", save_prefix=f"{outdir}/", save_suffix=f"{name}_{args.quality}_scatter.png") 

    plot_histogram(quality, gold_quality, args.quality, f"SWAPPED true&predicted for {args.quality} on {name}", save_prefix=f"{outdir}/", save_suffix=f"{name}_{args.quality}_histogram.png") 





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+")
    parser.add_argument("--dev", nargs="+")
    parser.add_argument("--test", nargs="+")
    parser.add_argument(
        "--include_quality",
        action="store_true",
        help="Adds the prediced quality to the embedding",
    )
    parser.add_argument("--n_bins", default=32, type=int)
    parser.add_argument("--dataloader_workers", default=0, type=int)
    parser.add_argument("--hid_dim", default=1024, type=int)
    parser.add_argument("--hid_layers", default=2, type=int)
    parser.add_argument("--batch_size", default=2048, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=5e-6, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument(
        "--precision",
        default=16,
        type=int,
        help="16 or 32, ATM we load 32 embeddings so 32 is needed",
    )
    parser.add_argument("--max_steps", default=10e6, type=int)
    parser.add_argument("--patience", default=3, type=int)
    parser.add_argument("--limit_train_batches", default=1.0, type=float)
    parser.add_argument("--limit_val_batches", default=1.0, type=float)
    parser.add_argument("--quality", default="appropriateness")
    parser.add_argument("--plda_feat_dim", default=None, type=int)
    parser.add_argument("--multiply_data", type=int, default=1)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument(
        "--add_gauss_noise",
        default=0.0,
        type=float,
        help="add gauss noise to avoid dependent rows for PCA decomposition. If PCA crashes with 0.0, start with 0.001.",
    )
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()
    seed_everything(args.seed, workers=True)
    with open(Path(args.dev[0]).parent / "args.json", "r") as f:
        d = json.load(f)
    dev_args = argparse.Namespace(**d)
    args.prompt = dev_args.prompt
    if args.outdir is None:
        args.outdir = exp_dirname(args, main_file=__file__)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    args.model_name = f"nn_ontop_{dev_args.wandb_run_id}"
    args.based_on = dev_args.wandb_run_id

    wandb_logger = pl.loggers.wandb.WandbLogger(
        entity="metric",
        project="chateval",
        save_dir=str(outdir),
    )
    # run_logger = get_wb_run_logger(outdir, vars(args))
    run_logger = wandb_logger.experiment

    with open(outdir / "args.json", "w") as f:
        args.wandb_run_id = run_logger.id
        d = vars(args)
        json.dump(d, f, indent=2)
        print(d, flush=True)
    symlink_target = (outdir.parent / run_logger.id).absolute()
    print(f"{symlink_target=}")
    os.symlink(outdir.relative_to(outdir.parent), symlink_target)

    limit_train_batches = (
        int(args.limit_train_batches)
        if args.limit_train_batches > 1.0
        else args.limit_train_batches
    )
    limit_val_batches = (
        int(args.limit_val_batches)
        if args.limit_val_batches > 1.0
        else args.limit_val_batches
    )
    ckpt_metric, ckpt_metric_mode = f"SRCC-{args.quality}", "max"
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        auto_insert_metric_name=False,
        save_last=False,
        mode=ckpt_metric_mode,
        monitor=ckpt_metric,
        save_top_k=2,
        # val_dataloader runs on whole epoch
    )
    trainer = pl.Trainer(
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        log_every_n_steps=2,
        gpus=args.gpus,
        max_steps=args.max_steps,
        detect_anomaly=False,
        precision=args.precision if args.gpus > 0 else 32,
        sync_batchnorm=True if args.gpus > 1 else False,
        strategy="ddp" if args.gpus > 1 else None,  # PL 1.5.+
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        replace_sampler_ddp=False,
        callbacks=[
            checkpoint_cb,
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.EarlyStopping(
                monitor=ckpt_metric,
                mode=ckpt_metric_mode,
                patience=args.patience,
            ),
        ],
        logger=wandb_logger,
    )
    train_ds = EmbDataset(args.train, args)
    dev_ds = EmbDataset(args.dev, args)
    test_ds = EmbDataset(args.test, args, has_gt=False) if args.test else None
    assert len(args.dev) == len(args.train), (len(args.dev), len(args.train))
    assert train_ds.input_size == dev_ds.input_size, (
        train_ds.input_size,
        dev_ds.input_size,
    )

    logging.warning(
        f"Ensembling {len(args.train)} models with {train_ds.input_size=}: {args.train}"
    )
    model = RegressionModule(args, train_ds.input_size)
    wandb_logger.watch(model)
    trainer.fit(
        model,
        train_dataloaders=DataLoader(
            train_ds,
            args.batch_size,
            num_workers=args.dataloader_workers,
            shuffle=True,
            drop_last=False,
        ),
        val_dataloaders=DataLoader(
            dev_ds,
            args.batch_size,
            num_workers=args.dataloader_workers,
            shuffle=False,
            drop_last=False,
        ),
    )

    best_model = RegressionModule.load_from_checkpoint(checkpoint_cb.best_model_path)

    predict(trainer, best_model, dev_ds, args, outdir, "dev")
    if args.test:
        predict(trainer, best_model, test_ds, args, outdir, "test")

    print(f"{symlink_target=}", flush=True)

