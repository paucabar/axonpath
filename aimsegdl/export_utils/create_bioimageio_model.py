import os
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from aimsegdl.inference.inference import load_model
from aimsegdl.pipeline import Pipeline
from aimsegdl.utils.image_processing import normalize

from bioimageio.spec.model.v0_5 import (
    ModelDescr, Author, LicenseId, RelativeFilePath,
    HttpUrl, InputTensorDescr, OutputTensorDescr, TensorId,
    AxisId, BatchAxis, ChannelAxis, SpaceInputAxis, SpaceOutputAxis,
    FileDescr, IntervalOrRatioDataDescr, SizeReference,
    ParameterizedSize, TorchscriptWeightsDescr, WeightsDescr,
    CiteEntry, Doi, Identifier
)
from bioimageio.spec import save_bioimageio_package
from bioimageio.core import test_model


def write_readme(model_name: str, output_dir: str) -> str:
    """
    Write a simple README file for the exported model.
    
    Returns the relative filename (to be used in the RDF).
    """
    readme_filename = model_name + "_README.md"
    readme_path = os.path.join(output_dir, readme_filename)

    with open(readme_path, "w") as f:
        f.write(f"# {model_name}\n")
        f.write("This model segments axons and fibres in EM images using AimSegDL.\n")
        f.write("Please refer to the AimSegDL documentation for inference and post-processing steps.\n")

    return readme_filename


def export_bioimageio(
    model_path: str,
    model_name: str,
    test_img_path: str,
    output_dir: str = "bioimageio_model"
):
    """
    Export AimSegDL TorchScript model to BioImage.IO package.

    Args:
        model_path (str): Path to .pth weights file.
        model_name (str): Output model name (used as .zip name and folder name).
        test_img_path (str): Path to a grayscale test image.
        output_dir (str): Directory to store exported model files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model and wrap in pipeline
    model = load_model(model_path, device="cpu")
    model.eval()
    wrapped_model = Pipeline(model)

    # Prepare test input
    img = np.array(Image.open(test_img_path)).astype(np.float32)
    input_ = normalize(img)[None, None]  # [1, 1, H, W]
    input_tensor = torch.from_numpy(input_)
    crop = input_tensor[:, :, :512, :512]  # Crop for scripting

    # Script and save TorchScript model
    scripted_model = torch.jit.script(wrapped_model, crop)
    torch.jit.save(scripted_model, os.path.join(output_dir, "weights.pt"))

    # Save test input and output
    np.save(os.path.join(output_dir, "test-input.npy"), crop.numpy())
    with torch.no_grad():
        output = scripted_model(crop)
    np.save(os.path.join(output_dir, "test-output.npy"), output.cpu().numpy())

    # Write README inside output_dir
    readme_filename = write_readme(model_name, output_dir)

    # Build RDF (Model Description)
    input_descr = InputTensorDescr(
        id=TensorId("raw"),
        axes=[
            BatchAxis(),
            ChannelAxis(id=AxisId("c"), channel_names=[Identifier("gray")]),
            SpaceInputAxis(id=AxisId("y"), size=ParameterizedSize(min=32, step=1)),
            SpaceInputAxis(id=AxisId("x"), size=ParameterizedSize(min=32, step=1)),
        ],
        data=IntervalOrRatioDataDescr(type="float32"),
        test_tensor=FileDescr(source="test-input.npy")
    )

    output_descr = OutputTensorDescr(
        id=TensorId("prediction"),
        axes=[
            BatchAxis(),
            ChannelAxis(
                id=AxisId("c"),
                channel_names=[
                    Identifier("semantic_class_index"),
                    Identifier("fibre_distance"),
                    Identifier("axon_distance")
                ]
            ),
            SpaceOutputAxis(id=AxisId("y"), size=SizeReference(tensor_id=TensorId("raw"), axis_id=AxisId("y"))),
            SpaceOutputAxis(id=AxisId("x"), size=SizeReference(tensor_id=TensorId("raw"), axis_id=AxisId("x"))),
        ],
        test_tensor=FileDescr(source="test-output.npy")
    )

    model_descr = ModelDescr(
        name=model_name,
        version="0.1.0",
        description="AimSegDL TorchScript model for axon/fibre segmentation.",
        authors=[Author(name="Pau Carrillo-Barberà")],
        license=LicenseId("CC-BY-4.0"),
        documentation=RelativeFilePath(Path(output_dir).name + "/" + readme_filename),
        covers=["cover.png"],  # optional; can skip if not present
        git_repo=HttpUrl("https://github.com/paucabar/aimseg-dl"),
        inputs=[input_descr],
        outputs=[output_descr],
        weights=WeightsDescr(
            torchscript=TorchscriptWeightsDescr(
                source=RelativeFilePath(Path(output_dir).name + "/" +"weights.pt"),
                pytorch_version=torch.__version__,
            )
        ),
        cite=[CiteEntry(text="Carrillo-Barberà et al., 2025", doi=Doi("10.1234/fake-doi-placeholder"))]  # TODO: update DOI
    )

    # Save model package
    zip_path = Path(output_dir) / f"{model_name}.zip"
    package_path = save_bioimageio_package(model_descr, output_path=zip_path)
    print("Saved model package:", package_path)

    # Validate RDF + package
    summary = test_model(model_descr)
    summary.display()
