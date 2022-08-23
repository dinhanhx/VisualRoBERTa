conda create --name torch-tpu python=3.8 pip
conda activate torch-tpu
pip3 install torch==1.11.0 https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.11-cp38-cp38-linux_x86_64.whl
pip3 install torch_xla[tpuvm]
pip3 install transformers[torch]