pip3 install torch==1.11.0
pip3 install torchvision==0.12.0

# TPU paackgages
pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.11-cp38-cp38-linux_x86_64.whl
pip3 install torch_xla[tpuvm]

pip3 install pandas
pip3 install einops
pip3 install sentencepiece
pip3 install transformers[torch]==4.21.1

# Lightning packages
pip3 install pytorch-lightning==1.6.5
pip3 install rich

# Evaluate packages
pip3 install evaluate
pip3 install pyvi
pip3 install nltk
pip3 install rouge_score