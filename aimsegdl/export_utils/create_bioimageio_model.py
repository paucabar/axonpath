import os
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from aimsegdl.inference.inference import load_model
from aimsegdl.pipeline import Pipeline
from aimsegdl.utils.image_processing import normalize

from bioimageio.spec.model.v0_5 import (
    ModelDescr,
    Author,
    LicenseId,
    RelativeFilePath,
    HttpUrl,
    InputTensorDescr,
    OutputTensorDescr,
    TensorId,
    AxisId,
    BatchAxis,
    ChannelAxis,
    SpaceInputAxis,
    SpaceOutputAxis,
    FileDescr,
    IntervalOrRatioDataDescr,
    SizeReference,
    ParameterizedSize,
    TorchscriptWeightsDescr,
    WeightsDescr,
    CiteEntry,
    Doi,
    Identifier,
    generate_covers,
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
    #crop = input_tensor[:, :, :512, :512]  # Crop for scripting

    # Script and save TorchScript model
    scripted_model = torch.jit.script(wrapped_model, input_tensor)
    torch.jit.save(scripted_model, os.path.join(output_dir, "weights.pt"))

    # Save test input and output
    np.save(os.path.join(output_dir, "test-input.npy"), input_tensor.detach().cpu().numpy())
    with torch.no_grad():
        output = scripted_model(input_tensor)

    # Ensure it's raw NumPy, not xarray or torch tensor
    output_np = output.detach().cpu().numpy().astype(np.float32)

    # Avoid fancy indexing issues — make sure it's a plain ndarray
    assert isinstance(output_np, np.ndarray) and output_np.ndim == 4

    np.save(os.path.join(output_dir, "test-output.npy"), output_np)

    print("input:", input_tensor.shape)
    print("output:", output.shape)
    print("output dtype:", output.dtype)
    print("Output sample values:", output_np.flatten()[::1000][:10])
    print("semantic unique:", np.unique(output[0:1, 0:1, :, :]), "dtype", output[0:1, 0:1, :, :].dtype)
    print("dt_fibre unique:", np.unique(output[0:1, 1:2, :, :]), "dtype", output[0:1, 1:2, :, :].dtype)
    print("dt_axon unique:", np.unique(output[0:1, 2:3, :, :]), "dtype", output[0:1, 2:3, :, :].dtype)
    print("test-input-shape", np.load("bioimageio_model/test-input.npy").shape)
    print("test-output-shape", np.load("bioimageio_model/test-output.npy").shape)

    # Confirm similarity
    expected = output.cpu().numpy()
    predicted = np.load("bioimageio_model/test-output.npy")

    np.testing.assert_allclose(expected, predicted, rtol=1e-3, atol=1e-3)
    assert input_tensor.shape[2:] == output.shape[2:], "Input/output shape mismatch!"


    # Write README inside output_dir
    readme_filename = write_readme(model_name, output_dir)

    # Model pixel size
    model_pixel_size = 0.008 # TODO: add as argument (temp fixed for EM-CNS)

    # Define input
    input_descr = InputTensorDescr(
        id=TensorId("raw"),
        axes=[
            BatchAxis(),
            ChannelAxis(id=AxisId("channel"), channel_names=[Identifier("gray")]),
            SpaceInputAxis(id=AxisId("y"), size=ParameterizedSize(min=512, step=1), scale=model_pixel_size, unit="micrometer"),
            SpaceInputAxis(id=AxisId("x"), size=ParameterizedSize(min=512, step=1), scale=model_pixel_size, unit="micrometer"),
        ],
        data=IntervalOrRatioDataDescr(type="float32"),
        test_tensor=FileDescr(source=os.path.join(output_dir, "test-input.npy"))
    )

    # Define output
    output_descr = OutputTensorDescr(
        id=TensorId("prediction"),
        axes=[
            BatchAxis(),
            ChannelAxis(
                id=AxisId("channel"),
                channel_names=[
                    Identifier("semantic_class_index"),
                    Identifier("fibre_distance"),
                    Identifier("axon_distance")
                ],
            ),
            SpaceOutputAxis(id=AxisId("y"), size=SizeReference(tensor_id=TensorId("raw"), axis_id=AxisId("y")), scale=model_pixel_size, unit="micrometer"),
            SpaceOutputAxis(id=AxisId("x"), size=SizeReference(tensor_id=TensorId("raw"), axis_id=AxisId("x")), scale=model_pixel_size, unit="micrometer"),
        ],
        test_tensor=FileDescr(source=os.path.join(output_dir, "test-output.npy"))
    )

    # Create an output descriptor that can be matched by a cover
    output_descr_cover = OutputTensorDescr(
        id=TensorId("prediction"),
        axes=[
            ChannelAxis(
                id=AxisId("channel"),
                channel_names=[Identifier("fibre_ditancemap")],  # only one channel now
            ),
            SpaceOutputAxis(id=AxisId("y"), size=SizeReference(tensor_id=TensorId("raw"), axis_id=AxisId("y")), scale=model_pixel_size, unit="micrometer"),
            SpaceOutputAxis(id=AxisId("x"), size=SizeReference(tensor_id=TensorId("raw"), axis_id=AxisId("x")), scale=model_pixel_size, unit="micrometer"),
        ],
        test_tensor=FileDescr(source=os.path.join(output_dir, "test-output.npy"))
)

    # Generate a cover for the bioimageio model
    covers = generate_covers(
        inputs=[(input_descr, input_)],
        outputs=[(output_descr_cover, output_np[0, 1:2, :, :])]
    )

    # Save the first cover
    cover_path = os.path.join(output_dir, "cover.png")

    import shutil
    shutil.copy(covers[0], cover_path)


    #Define model
    model_descr = ModelDescr(
        name=model_name,
        version="0.1.0",
        description="AimSegDL TorchScript model for axon/fibre segmentation.",
        authors=[Author(name="Pau Carrillo-Barberà")], # TODO: update author list
        license=LicenseId("CC-BY-4.0"),
        documentation=RelativeFilePath(Path(output_dir).name + "/" + readme_filename),
        covers=[cover_path],
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
    summary = test_model(model_descr, weight_format="torchscript", test_tolerance=1e-2)
    summary.display()
