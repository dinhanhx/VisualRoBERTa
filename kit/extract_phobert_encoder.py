import torch
from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM

if __name__ == "__main__":
    model_name = "phobert-base"
    phobert_model = RobertaForMaskedLM.from_pretrained(f"vinai/{model_name}")
    torch.save(
        phobert_model.roberta.encoder.state_dict(), f"assets/{model_name}-encoder.pt"
    )
