# TPU VM 3.8

Run these commands before running any process with TPU
```bash
export TPU_LOG_DIR="disabled"
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export PT_XLA_DEBUG=1
export USE_TORCH=ON
```

## Conda env setup

To create a simple env
```bash
conda create --name torch-tpu python=3.8 pip
conda activate torch-tpu
```

Inside `torch-tpu`, simply run `bash install_cmds.sh`