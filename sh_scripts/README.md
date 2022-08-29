# TPU VM 3.8

Run these commands before running any process with TPU
```bash
export TPU_LOG_DIR="disabled"
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export PT_XLA_DEBUG=1
export USE_TORCH=ON
```

`env_setup.sh` to setup a conda environment which has
- torch with tpu support
- torchvision
- transformer[torch]

`install_cmds.sh` like `env_setup.sh` but just to do `pip install`