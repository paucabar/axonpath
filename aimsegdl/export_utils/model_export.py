import torch
import torch.nn as nn
from aimsegdl.pipeline import Pipeline



def export_torchscript_model(
    model: nn.Module,
    export_path: str,
    example_input: torch.Tensor = None,
    device: str = "cpu"
):
    """
    Export the given model using a TorchScript pipeline wrapper.

    Parameters:
        model (nn.Module): Trained PyTorch model.
        export_path (str): Path to save the exported .pt file.
        example_input (torch.Tensor, optional): Example input for tracing/scripting.
        device (str): Device to perform scripting on.
    """
    model.eval()
    model.to(device)

    scripted_pipeline = Pipeline(model)

    if example_input is None:
        # Default dummy input (1 channel, 512x512)
        example_input = torch.zeros(1, 1, 512, 512).to(device)

    scripted_model = torch.jit.script(scripted_pipeline, example_input)
    torch.jit.save(scripted_model, export_path)
    print(f"TorchScript model saved to: {export_path}")