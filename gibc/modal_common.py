"""Modal image and volumes shared by modal_data.py, modal_train.py and modal_eval.py."""
import modal

from gibc.configs import DATA_DIR, DATA_VOL, RUNS_DIR, RUNS_VOL

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.11.0",
        "tokenizers==0.23.2",
        "transformers==5.17.0",
        "lm_eval[hf]==0.4.13",
        "pyarrow>=17",
        "huggingface_hub>=0.30",
        "numpy>=2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "TOKENIZERS_PARALLELISM": "true"})
    .add_local_python_source("gibc")  # must stay last: local files are mounted at start
)

data_vol = modal.Volume.from_name(DATA_VOL, create_if_missing=True)
runs_vol = modal.Volume.from_name(RUNS_VOL, create_if_missing=True)
VOLUMES = {DATA_DIR: data_vol, RUNS_DIR: runs_vol}

hf_secret = modal.Secret.from_name("huggingface")
H100 = "H100!"  # pin to H100 so throughput numbers are comparable across runs
