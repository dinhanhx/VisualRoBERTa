from pathlib import Path

import torch

from exp.run_pretrain import Wrapper

if __name__ == "__main__":
    ckpt_folder = Path("logs/lightning_logs/version_0/checkpoints")
    ckpt_file = ckpt_folder.joinpath("epoch=1-step=18000.ckpt")
    assert ckpt_file.is_file()
    wrapper = Wrapper.load_from_checkpoint(checkpoint_path=ckpt_file)
    pt_file = ckpt_folder.joinpath("imagetext-base.pt")
    torch.save(wrapper.model.state_dict(), pt_file)
