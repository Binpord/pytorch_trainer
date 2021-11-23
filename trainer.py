import os

import matplotlib.pyplot as plt
import torch

from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Any

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset, DataLoader


class Logger:
    def log(self, data: Dict[str, Any]) -> None:
        raise NotImplementedError()


class PrintLogger(Logger):
    def log(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            print(key, value)


# For the sake of not having wandb installed, I am not writing
# WandbLogger class, however note that you can pass wandb module
# itself into learner's logger paramenter and will work just fine.


class Metric:
    def update(self, prediction: Tensor, target: Tensor) -> None:
        raise NotImplementedError()

    def compute(self) -> float:
        raise NotImplementedError()

    def reset(self) -> None:
        raise NotImplementedError()


class Accuracy(Metric):
    def update(self, prediction: Tensor, target: Tensor) -> None:
        prediction = torch.argmax(prediction, dim=1)
        self.correct += torch.sum(prediction == target)
        self.total += torch.numel(target)

    def compute(self) -> float:
        return self.correct / self.total

    def reset(self) -> None:
        self.correct = 0
        self.total = 0


class TrainableModel(Module):
    def __init__(self, criterion: _Loss) -> None:
        super().__init__()
        self.criterion = criterion

    def training_step(self, input: Tensor, target: Tensor) -> Tensor:
        prediction = self(input)
        loss = self.criterion(prediction, target)
        return loss

    def validation_step(self, input, target) -> Tuple[Tensor, Tensor]:
        prediction = self(input)
        loss = self.criterion(prediction, target)
        return loss, prediction

    def configure_optimizers(
        self, learning_rate: float, n_epochs: int, steps_per_epoch: int
    ) -> Tuple[Optimizer, _LRScheduler]:
        optimizer = torch.optim.AdamW(self.parameters(), learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            learning_rate,
            epochs=n_epochs,
            steps_per_epoch=steps_per_epoch,
        )
        return optimizer, scheduler


@dataclass
class LearningRateFinderResults:
    learning_rate_history: List[float]
    loss_history: List[float]

    def plot_results(
        self, n_skip_last: Optional[int] = None, ax: Optional[plt.Axes] = None
    ) -> None:
        if ax is None:
            ax = plt.gca()

        slice_stop = -n_skip_last if n_skip_last is not None else None
        ax.plot(
            self.learning_rate_history[:slice_stop],
            self.loss_history[:slice_stop],
        )
        ax.set_xscale("log")


class Trainer:
    def __init__(
        self,
        model: TrainableModel,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 64,
        n_workers: Optional[int] = None,
        metrics: Optional[List[Metric]] = None,
        gpu_number: int = 0,
        logger: Optional[Logger] = None,
    ) -> None:
        self.model = model
        self.metrics = metrics or []
        self.device = torch.device(
            f"cuda:{gpu_number}" if torch.cuda.is_available() else "cpu"
        )
        self.logger = logger or PrintLogger()

        self.train_dataloader, self.val_dataloader = self.configure_dataloaders(
            train_dataset, val_dataset, batch_size, n_workers
        )
        self.model.to(self.device)
        self.optimizer = None
        self.scheduler = None

    def train_model(self, n_epochs: int = 10, learning_rate: float = 1e-3) -> None:
        self.optimizer, self.scheduler = self.model.configure_optimizers(
            learning_rate, n_epochs, steps_per_epoch=len(self.train_dataloader)
        )
        for epoch in range(n_epochs):
            self.training_epoch()

            # Validate if the validation dataset is available.
            if self.val_dataloader is None:
                continue

            with torch.no_grad():
                val_loss, metric_values = self.validation_epoch()

            val_summary = {
                "epoch": epoch,
                "val_loss": val_loss,
            }
            val_summary.update(metric_values)
            self.logger.log(val_summary)

    def training_epoch(self) -> None:
        self.model.train()
        for input, target in self.train_dataloader:
            input, target = input.to(self.device), target.to(self.device)
            loss = self.model.training_step(input, target)
            self.optimization_step(loss)
            self.logger.log({"train_loss": loss.item()})

    def optimization_step(self, loss: Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def validation_epoch(self) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        self.reset_metrics()
        val_loss = 0
        for input, target in self.val_dataloader:
            input, target = input.to(self.device), target.to(self.device)
            loss, prediction = self.model.validation_step(input, target)
            val_loss += loss.item()
            self.update_metrics(prediction, target)

        return val_loss / len(self.val_dataloader), self.compute_metrics()

    def configure_dataloaders(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 64,
        n_workers: Optional[int] = None,
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        if n_workers is None:
            n_workers = os.cpu_count()

        train_dataloader = DataLoader(
            train_dataset,
            batch_size,
            shuffle=True,
            num_workers=n_workers,
        )

        val_dataloader = None
        if val_dataset is not None:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size,
                shuffle=False,
                num_workers=n_workers,
            )

        return train_dataloader, val_dataloader

    def find_learning_rate(
        self,
        start_value: float = 1e-7,
        end_value: float = 10.0,
        n_steps: int = 100,
        beta: float = 0.98,
        early_stop: bool = True,
    ) -> LearningRateFinderResults:
        self.save_model()

        gamma = (end_value / start_value) ** (1 / n_steps)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=start_value)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma)
        self.model.train()

        learning_rate_history, loss_history = [], []
        average_loss = 0.0
        best_loss = None
        train_dataloader = iter(self.train_dataloader)
        for step in range(n_steps):
            # In case of len(train_dataloader) < n_steps.
            try:
                input, target = next(train_dataloader)
            except StopIteration:
                train_dataloader = iter(self.train_dataloader)
                input, target = next(train_dataloader)

            loss = self.model.training_step(
                input.to(self.device), target.to(self.device)
            )

            # Apply exponential smoothing.
            average_loss = average_loss * beta + loss.item() * (1 - beta)
            smooth_loss = average_loss / (1 - beta ** (step + 1))
            if step == 0 or smooth_loss < best_loss:
                best_loss = smooth_loss
            elif early_stop and step > 0 and smooth_loss > 4 * best_loss:
                break

            learning_rate_history.append(self.optimizer.param_groups[0]["lr"])
            loss_history.append(smooth_loss)

            self.optimization_step(loss)

        self.load_model()
        return LearningRateFinderResults(learning_rate_history, loss_history)

    def save_model(self, path="model.pth") -> None:
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="model.pth") -> None:
        self.model.load_state_dict(torch.load(path))

    def reset_metrics(self) -> None:
        for metric in self.metrics:
            metric.reset()

    def update_metrics(self, prediction: Tensor, target: Tensor) -> None:
        for metric in self.metrics:
            metric.update(prediction, target)

    def compute_metrics(self) -> Dict[str, float]:
        return {
            f"val_{self.metric_name(metric)}": metric.compute()
            for metric in self.metrics
        }

    def metric_name(self, metric: Metric) -> str:
        return metric.__class__.__name__.lower()
