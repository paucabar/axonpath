import torch
import torch.nn as nn
from aimsegdl.pipeline import Pipeline



def export_torchscript_model(
    model: nn.Module,
    export_path: str,
    device: str = "cpu"
):
    """
    Export the given model using a TorchScript pipeline wrapper.

    Parameters:
        model (nn.Module): Trained PyTorch model.
        export_path (str): Path to save the exported .pt file.
        device (str): Device to perform scripting on.
    """
    model.eval()
    model.to(device)

    scripted_pipeline = Pipeline(model)
    scripted_model = torch.jit.script(scripted_pipeline)
    torch.jit.save(scripted_model, export_path)
    print(f"TorchScript model saved to: {export_path}")