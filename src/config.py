"""GCIM-MSD data paths and training defaults."""

import os
from pathlib import Path


DATA_ROOT = Path(
    os.environ.get(
        "GCIM_MSD_DATA_ROOT",
        Path(__file__).resolve().parents[1] / "data",
    )
).expanduser().resolve()

# The dataset loader concatenates root_dir and relative paths.
root_dir = f"{DATA_ROOT}{os.sep}"

######## Memotion ########
me_train_csv_name = "memotion/splits/train.csv"
me_dev_csv_name = "memotion/splits/dev.csv"
me_test_csv_name = "memotion/splits/test.csv"
me_image_vec_dir = "memotion/features/image/"
me_text_vec_dir = "memotion/features/text/"

######## MUStARD ########
mu_train_csv_name = "mustard/splits/train.csv"
mu_dev_csv_name = "mustard/splits/dev.csv"
mu_test_csv_name = "mustard/splits/test.csv"
mu_image_vec_dir = "mustard/features/image/"
mu_text_vec_dir = "mustard/features/text/"
mu_audio_vec_dir = "mustard/features/audio/"

######## Training defaults ########
batch_size = 32
epochs = 30
lr = 2e-4
gradient_accumulation_steps = 2
