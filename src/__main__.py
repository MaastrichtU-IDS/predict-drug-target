import typer

from src.train import compute_and_train

cli = typer.Typer()


@cli.command()
def train(
    known_drug_target: str = typer.Argument(..., help="Input drug-target CSV file to train the model"),
    output: str = typer.Option("output", "-o", help="Output directory to save the model"),
):
    """Train a model with input file and save output in the specified directory.

    Args:
      file (str): Input file for model training.
      output_dir (str): Output directory to save the model.

    Examples:
      $ predict-dt train known_drug_target.csv -o data/my_model
    """
    scores = compute_and_train(known_drug_target, output)
    typer.echo(f"Training done: {scores}")


@cli.command()
def version():
    """Display the package version."""
    typer.echo("0.0.1")


if __name__ == "__main__":
    cli()
