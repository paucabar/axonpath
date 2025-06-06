import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import monai
from utils.image_processing import normalize, apply_semantic_segmentation_head_scriptable

# the imports for bioimage.io model export
import bioimageio.core
from bioimageio.core.build_spec import build_model
from bioimageio.core.resource_tests import test_model



import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.image_processing import apply_semantic_segmentation_head_scriptable


class Pipeline(nn.Module):
    """
    TorchScript-compatible inference pipeline.

    This wraps a model and handles input padding, semantic head application,
    and output reassembly.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.target_height = 512
        self.target_width = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with padding, inference, semantic head, and unpadding.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor after postprocessing (1, C_out, H, W)
        """
        b, c, h, w = x.shape
        pad_h = max(0, self.target_height - h)
        pad_w = max(0, self.target_width - w)

        # Pad input
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant')

        # Model forward
        pred = self.model(x)

        # Semantic segmentation postprocessing
        semantic = apply_semantic_segmentation_head_scriptable(pred[:, 0:3, :, :])
        dt_fibre = pred[:, 3, :, :]
        dt_axon = pred[:, 4, :, :]

        # Combine all outputs
        output = torch.cat((semantic, dt_fibre, dt_axon), dim=0).unsqueeze(0)

        # Remove padding
        output = output[:, :, :h, :w]

        return output.float()



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


def readme(model_name: str):
    # create markdown documentation for your model
    # this should describe how the model was trained, (and on which data)
    # and also what to take into consideration when running the model, especially how to validate the model
    # here, we just create a stub documentation
    with open(os.path.join(model_name, model_name + "_README.md"), "w") as f:
        f.write("# My First Model\n")
        f.write("This model was trained on a very big dataset.\n")
        f.write("You should not let it get wet or feed it after midnight.\n")
        f.write("To validate its predictins, make sure that it does not produce any evil clones.\n")


def export_bioimageio(model: monai.networks.nets.unet.UNet, model_name: str, deepimagej: bool, test_img_path: str):
    # create a temporary directory to store intermediate files
    os.makedirs(model_name, exist_ok=True)

    model.eval()
    device = "cpu"
    model.to(device)

    # import test data
    input_ = np.array(Image.open(test_img_path)).astype(np.float32)
    input_ = normalize(input_)[(None,)*2] # unsqueeze(0) twice
    data = torch.tensor((input_))#.to(device)
    print(data.shape)

    # export to torchscript and save the model weights
    crop = data[:,:,:512,:512]#.to(device)
    my_pipeline = Pipeline(model)
    model = torch.jit.script(my_pipeline, crop.to(device))
    torch.jit.save(model, os.path.join(model_name, "weights.pt"))

    # create test data for this model: an input image and an output image
    # this data will be used for model test runs to ensure the model runs correctly and that the expected output can be reproduced
    # NOTE: if you have pre-and-post-processing in your model (see the more advanced models for an example)
    # you will need to save the input BEFORE preprocessing and the output AFTER postprocessing

    np.save(os.path.join(model_name, "test-input.npy"), crop)

    with torch.no_grad():
        output = model(crop.to(device))
        print(output.shape)
    np.save(os.path.join(model_name, "test-output.npy"), output)

    # create readme
    readme(model_name)

    # now we can use the build_model function to create the zipped package.
    # it takes the path to the weights and data we have just created, as well as additional information
    # that will be used to add metadata to the rdf.yaml file in the model zip
    # we only use a subset of the available options here, please refer to the advanced examples and to the
    # function signature of build_model in order to get an overview of the full functionality
    _ = build_model(
        # the weight file and the type of the weights
        weight_uri = os.path.join(model_name, "weights.pt"),
        weight_type = "torchscript",
        # the test input and output data
        test_inputs = [os.path.join(model_name, "test-input.npy")],
        test_outputs = [os.path.join(model_name, "test-output.npy")],
        # where to save the model zip, how to call the model and a short description of it
        output_path = os.path.join(model_name, model_name + ".zip"),
        name = model_name,
        description = "a fancy new model",
        # additional metadata about authors, licenses, citation etc.
        authors = [{"name": "Pau Carrillo-Barberà"}],
        license = "CC-BY-4.0",
        documentation = os.path.join(model_name, model_name + "_README.md"),
        tags = ["axon-segmentation"],  # the tags are used to make models more findable on the website
        cite = [{"text": "Carrillo-Barberà et al.", "doi": "TODO"}],
        # description of the tensors
        # these are passed as list because we support multiple inputs / outputs per model
        input_names = ["raw"],
        input_axes = ["bcyx"],
        input_min_shape = [[1, 1, 512, 512]],
        input_step = [[0, 0, 256, 256]],
        output_names = ["semantic"],
        output_axes=["bcyx"],
        output_reference = ["raw"],
        output_scale = [[1.0, 3.0, 1.0, 1.0]],
        output_offset = [[0.0, 0.0, 0.0, 0.0]],
        preprocessing = None,
        add_deepimagej_config = deepimagej,
    )

    # finally, we test that the expected outptus are reproduced when running the model.
    # the 'test_model' function runs this test.
    # it will output a list of dictionaries. each dict gives the status of a different test that is being run
    # if all of them contain "status": "passed" then all tests were successful
    my_model = bioimageio.core.load_resource_description(os.path.join(model_name, model_name + ".zip")) 
    test_model(my_model)