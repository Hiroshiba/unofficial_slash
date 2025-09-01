"""機械学習モデルの学習メインスクリプト"""

import argparse
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from unofficial_slash.batch import BatchOutput, collate_dataset_output
from unofficial_slash.config import Config, DataFileConfig
from unofficial_slash.dataset import (
    Dataset,
    DatasetType,
    create_dataset,
    prefetch_datas,
)
from unofficial_slash.evaluator import (
    Evaluator,
    EvaluatorOutput,
    calculate_value,
)
from unofficial_slash.generator import Generator
from unofficial_slash.model import Model, ModelOutput
from unofficial_slash.network.predictor import create_predictor
from unofficial_slash.sampler import LengthBatchSampler
from unofficial_slash.utility.pytorch_utility import (
    detach_cpu,
    init_weights,
    make_optimizer,
    make_scheduler,
)
from unofficial_slash.utility.train_utility import (
    DataNumProtocol,
    Logger,
    SaveManager,
    reduce_result,
)


def _delete_data_num(output: DataNumProtocol) -> dict[str, Any]:
    if not hasattr(output, "data_num"):
        raise ValueError("Output does not have 'data_num' attribute")
    return {k: v for k, v in asdict(output).items() if k != "data_num"}


@dataclass
class TrainingResults:
    """学習結果"""

    train: ModelOutput

    def to_summary_dict(self) -> dict[str, Any]:
        """ログ出力用の辞書を生成"""
        return {DatasetType.TRAIN.value: _delete_data_num(self.train)}


@dataclass
class EvaluationResults:
    """評価結果"""

    test: ModelOutput
    eval: EvaluatorOutput | None
    valid: EvaluatorOutput | None

    def to_summary_dict(self) -> dict[str, Any]:
        """ログ出力用の辞書を生成"""
        summary = {
            DatasetType.TEST.value: _delete_data_num(self.test),
        }
        if self.eval is not None:
            summary[DatasetType.EVAL.value] = _delete_data_num(self.eval)
        if self.valid is not None:
            summary[DatasetType.VALID.value] = _delete_data_num(self.valid)
        return summary


@dataclass
class TrainingContext:
    """学習に必要な全てのオブジェクトをまとめる"""

    config: Config
    train_loader: DataLoader
    test_loader: DataLoader
    eval_loader: DataLoader | None
    valid_loader: DataLoader | None
    model: Model
    evaluator: Evaluator
    optimizer: torch.optim.Optimizer
    scaler: GradScaler
    logger: Logger
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    save_manager: SaveManager
    device: str
    epoch: int
    iteration: int
    snapshot_path: Path


def create_data_loader(
    config: Config, data_file_config: DataFileConfig, dataset: Dataset, for_train: bool
) -> DataLoader:
    """動的バッチング対応DataLoaderを作成"""
    batch_sampler = LengthBatchSampler(
        batch_bins=data_file_config.batch_bins,
        lengths=dataset.lengths,
        min_batch_size=config.train.min_batch_size,
        drop_last=for_train,
    )
    return DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=config.train.preprocess_workers,
        collate_fn=collate_dataset_output,
        pin_memory=config.train.use_gpu,
        timeout=0 if config.train.preprocess_workers == 0 else 30,
        persistent_workers=config.train.preprocess_workers > 0,
    )


def setup_training_context(config_yaml_path: Path, output_dir: Path) -> TrainingContext:
    """TrainingContextを作成"""
    # config
    with config_yaml_path.open() as f:
        config_dict = yaml.safe_load(f)
    config = Config.from_dict(config_dict)
    config.add_git_info()
    config.validate_config()

    # dataset
    datasets = create_dataset(config.dataset)

    # prefetch
    datas = datasets.train.datas + datasets.test.datas
    datas += datasets.eval.datas if datasets.eval is not None else []
    datas += datasets.valid.datas if datasets.valid is not None else []
    threading.Thread(
        target=prefetch_datas,
        args=(datas, config.train.prefetch_workers, config.dataset.sample_rate),
        daemon=True,
    ).start()

    # data loader
    train_loader = create_data_loader(
        config, config.dataset.train, datasets.train, for_train=True
    )
    test_loader = create_data_loader(
        config, config.dataset.train, datasets.test, for_train=False
    )
    eval_loader = (
        create_data_loader(config, config.dataset.train, datasets.eval, for_train=False)
        if datasets.eval is not None
        else None
    )
    valid_loader = (
        create_data_loader(
            config, config.dataset.valid, datasets.valid, for_train=False
        )
        if datasets.valid is not None and config.dataset.valid is not None
        else None
    )

    # predictor
    predictor = create_predictor(config.network)
    device = "cuda" if config.train.use_gpu else "cpu"
    if config.train.pretrained_predictor_path is not None:
        state_dict = torch.load(
            config.train.pretrained_predictor_path, map_location=device
        )
        predictor.load_state_dict(state_dict)
    print("predictor:", predictor)

    # model
    # FIXME: TorchScript化を一時的に無効化（nnAudio/CQTがtorch.jit.scriptと互換性がないため）
    # predictor_scripted: Predictor = torch.jit.script(predictor)  # type: ignore
    predictor_scripted = predictor  # TorchScript化をスキップ
    model = Model(model_config=config.model, predictor=predictor_scripted)
    if config.train.weight_initializer is not None:
        init_weights(model, name=config.train.weight_initializer)
    model.to(device)

    # evaluator
    # FIXME: TorchScript無効化に伴いpredictor（元のオブジェクト）を直接使用
    generator = Generator(
        config=config, predictor=predictor, use_gpu=config.train.use_gpu
    )
    evaluator = Evaluator(generator=generator)

    # optimizer
    optimizer = make_optimizer(config_dict=config.train.optimizer, model=model)
    scaler = GradScaler(device, enabled=config.train.use_amp)

    # logger
    logger = Logger(
        config_dict=config_dict,
        project_category=config.project.category,
        project_name=config.project.name,
        output_dir=output_dir,
    )

    # scheduler
    scheduler = None
    if config.train.scheduler is not None:
        scheduler = make_scheduler(
            config_dict=config.train.scheduler, optimizer=optimizer
        )

    # save
    save_manager = SaveManager(
        predictor=predictor,
        prefix="predictor_",
        output_dir=output_dir,
        top_num=config.train.model_save_num,
        last_num=config.train.model_save_num,
    )

    return TrainingContext(
        config=config,
        train_loader=train_loader,
        test_loader=test_loader,
        eval_loader=eval_loader,
        valid_loader=valid_loader,
        model=model,
        evaluator=evaluator,
        optimizer=optimizer,
        scaler=scaler,
        logger=logger,
        scheduler=scheduler,
        save_manager=save_manager,
        device=device,
        epoch=0,
        iteration=0,
        snapshot_path=output_dir / "snapshot.pth",
    )


def load_snapshot(context: TrainingContext) -> None:
    """学習状態を復元"""
    snapshot = torch.load(context.snapshot_path, map_location=context.device)

    context.model.load_state_dict(snapshot["model"])
    context.optimizer.load_state_dict(snapshot["optimizer"])
    context.scaler.load_state_dict(snapshot["scaler"])
    context.logger.load_state_dict(snapshot["logger"])

    context.iteration = snapshot["iteration"]
    context.epoch = snapshot["epoch"]

    if context.scheduler is not None:
        context.scheduler.last_epoch = context.epoch


def train_one_epoch(context: TrainingContext) -> TrainingResults:
    """１エポックの学習処理"""
    context.model.train()
    if hasattr(context.optimizer, "train"):
        context.optimizer.train()  # type: ignore

    batch: BatchOutput
    train_results: list[ModelOutput] = []
    for batch in context.train_loader:
        context.iteration += 1

        with autocast(context.device, enabled=context.config.train.use_amp):
            batch = batch.to_device(context.device, non_blocking=True)
            result: ModelOutput = context.model(batch)

        loss = result.loss
        if loss.isnan():
            raise ValueError("loss is NaN")

        context.optimizer.zero_grad()
        context.scaler.scale(loss).backward()
        context.scaler.step(context.optimizer)
        context.scaler.update()

        train_results.append(detach_cpu(result))

    if context.scheduler is not None:
        context.scheduler.step()

    return TrainingResults(train=reduce_result(train_results))


@torch.no_grad()
def evaluate(context: TrainingContext) -> EvaluationResults:
    """評価値を計算する"""
    context.model.eval()
    if hasattr(context.optimizer, "eval"):
        context.optimizer.eval()  # type: ignore

    batch: BatchOutput

    # test評価
    test_result_list: list[ModelOutput] = []
    for batch in context.test_loader:
        batch = batch.to_device(context.device, non_blocking=True)
        result = context.model(batch)
        test_result_list.append(detach_cpu(result))
    test_result = reduce_result(test_result_list)

    # eval評価
    eval_result: EvaluatorOutput | None = None
    eval_result_list: list[EvaluatorOutput] = []
    if context.eval_loader is not None:
        for batch in context.eval_loader:
            batch = batch.to_device(context.device, non_blocking=True)
            result = context.evaluator(batch)
            eval_result_list.append(detach_cpu(result))
        eval_result = reduce_result(eval_result_list)

    # valid評価
    valid_result: EvaluatorOutput | None = None
    if context.valid_loader is not None:
        valid_result_list: list[EvaluatorOutput] = []
        for batch in context.valid_loader:
            batch = batch.to_device(context.device, non_blocking=True)
            result = context.evaluator(batch)
            valid_result_list.append(detach_cpu(result))
        valid_result = reduce_result(valid_result_list)

    return EvaluationResults(test=test_result, eval=eval_result, valid=valid_result)


def save_predictor(
    context: TrainingContext, evaluation_results: EvaluationResults
) -> None:
    """評価結果に基づいてPredictorを保存する"""
    if evaluation_results.valid is not None:
        evaluation_value = calculate_value(evaluation_results.valid).item()
    else:
        evaluation_value = 0

    context.save_manager.save(value=evaluation_value, step=context.epoch)


def save_checkpoint(context: TrainingContext) -> None:
    """チェックポイント保存する"""
    torch.save(
        {
            "model": context.model.state_dict(),
            "optimizer": context.optimizer.state_dict(),
            "scaler": context.scaler.state_dict(),
            "logger": context.logger.state_dict(),
            "iteration": context.iteration,
            "epoch": context.epoch,
        },
        context.snapshot_path,
    )


def should_log_epoch(context: TrainingContext) -> bool:
    """ログ出力するかどうか判定する"""
    return context.epoch % context.config.train.log_epoch == 0


def should_eval_epoch(context: TrainingContext) -> bool:
    """評価実行するかどうか判定する"""
    return context.epoch % context.config.train.eval_epoch == 0


def should_snapshot_epoch(context: TrainingContext) -> bool:
    """スナップショット保存するかどうか判定する"""
    return context.epoch % context.config.train.snapshot_epoch == 0


def training_loop(context: TrainingContext) -> None:
    """学習ループ"""
    for _ in range(context.config.train.stop_epoch):
        context.epoch += 1
        if context.epoch > context.config.train.stop_epoch:
            break

        training_results = train_one_epoch(context)

        if should_log_epoch(context):
            summary = {
                "iteration": context.iteration,
                "lr": context.optimizer.param_groups[0]["lr"],
            }
            summary.update(training_results.to_summary_dict())

            if should_eval_epoch(context):
                evaluation_results = evaluate(context)
                summary.update(evaluation_results.to_summary_dict())
                save_predictor(context, evaluation_results)

            context.logger.log(summary=summary, step=context.epoch)

        if should_snapshot_epoch(context):
            save_checkpoint(context)


def train(config_yaml_path: Path, output_dir: Path) -> None:
    """機械学習モデルを学習する。スナップショットがあれば再開する。"""
    context = setup_training_context(config_yaml_path, output_dir)

    if context.snapshot_path.exists():
        load_snapshot(context)

    output_dir.mkdir(exist_ok=True, parents=True)
    with (output_dir / "config.yaml").open(mode="w") as f:
        yaml.safe_dump(context.config.to_dict(), f)

    try:
        training_loop(context)
    finally:
        context.logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_yaml_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    train(**vars(parser.parse_args()))
