from pathlib import Path

import click

from .datasets import Dataset
from .networks.cnn import CNNRunner

_datasets_map = {
    "combined": Dataset,
}

_runners_map = {
    "cnn": CNNRunner,
}


@click.command()
@click.option("-e", "--num-epochs", default=100, help="Number of epochs.", required=False, type=int)
@click.option("-lr", "--learning-rate", default=0.01, help="Learning rate.", required=False, type=float)
@click.option(
    "-d",
    "--dataset",
    default=list(_datasets_map.keys())[0],
    help="Dataset to use.",
    required=False,
    type=click.Choice(list(_datasets_map.keys()), case_sensitive=False),
)
@click.option(
    "-M",
    "--model",
    default=list(_runners_map.keys())[0],
    help="Neural network model to use.",
    required=False,
    type=click.Choice(list(_runners_map.keys()), case_sensitive=False),
)
@click.option(
    "-m",
    "--momentum",
    default=0.99,
    help="SGD Momentum.",
    required=False,
    type=float,
)
@click.option(
    "-g",
    "--num-gates",
    default=1,
    help="Number of gates to use (for GRU).",
    required=False,
    type=int,
)
@click.option(
    "-cp",
    "--checkpoint",
    help="Checkpoint file to use.",
    required=False,
    type=str,
)
def main(num_epochs, learning_rate, dataset, model, momentum, num_gates, checkpoint):
    dataset_obj = _datasets_map[dataset]()
    train_loader, validation_loader, test_loader = dataset_obj.process()

    print(
        "CLI Args: "
        f"num_epochs={num_epochs}; "
        f"learning_rate={learning_rate}; "
        f"momentum={momentum}; "
        f"dataset={dataset}; "
        f"model={model}; "
        f"num_gates={num_gates}"
    )

    runner_kwargs = {
        "in_features": dataset_obj.in_features,
        "label_mappings": dataset_obj.get_label_mappings(),
        "learning_rate": learning_rate,
        "momentum": momentum,
        "num_epochs": num_epochs,
        "out_features": dataset_obj.out_features,
    }

    if model in ["gru", "gru_deep_ae", "gru_shallow_ae"]:
        runner_kwargs["num_gates"] = num_gates

    runner = _runners_map[model](**runner_kwargs)

    if checkpoint:
        checkpoint_filename = checkpoint
    else:
        checkpoint_filename = f"{runner.__class__.__name__}_checkpoint.ckpt"

    # if Path(checkpoint_filename).is_file():
    #     runner.load_state(checkpoint_filename)

    try:
        runner.train(train_loader, validation_loader)
    except KeyboardInterrupt:
        ...

    runner.save_state(checkpoint_filename)

    # runner.evaluate(test_loader)

    # runner.save_loss_plot()

    # runner.predict(final_test_set)
