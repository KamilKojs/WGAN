from src.config import read
from src.models.models import train
from typing import Dict
from pytorch_lightning import seed_everything

def main(
    seed: int,
    output_dir: str,
    data_args: Dict,
    model_args: Dict,
    trainer_args: Dict,
    early_stopping_args: Dict,
):
    seed_everything(seed)
    train(
        data_args,
        model_args,
        trainer_args,
        early_stopping_args,
        output_dir,
    )


if __name__ == "__main__":
    main(**read())