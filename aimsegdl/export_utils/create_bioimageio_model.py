import os
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import matplotlib.cm as cm
from scipy.ndimage import binary_fill_holes

from aimsegdl.inference.inference import load_model
from aimsegdl.pipeline import Pipeline
from aimsegdl.utils.image_processing import normalize, segment_instances_from_sdt, map_axon_labels_to_fibres
from aimsegdl.export_utils.config import BioimageioExportConfig
from aimsegdl.utils.visualization import get_glasbey_cmap

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
)

from bioimageio.spec import save_bioimageio_package
from bioimageio.core import test_model


def write_readme(config_bioimageio: BioimageioExportConfig) -> str:
    """
    Write a simple README file for the exported model.
    
    Returns the relative filename (to be used in the RDF).
    """
    readme_filename = "README.md"
    readme_path = os.path.join(config_bioimageio.output_dir, readme_filename)

    with open(readme_path, "w", encoding="utf-8") as f:
        # Title
        f.write(f"# {config_bioimageio.model_name}-{config_bioimageio.model_version}\n\n")

        # Purpose
        f.write("This model segments axons and fibres using the AimSegDL framework.\n\n")

        # Version & Licensing
        f.write(f"**Version**: {config_bioimageio.model_version}\n\n")
        f.write(f"**License**: {config_bioimageio.license_id}\n\n")

        # Pixel size
        f.write(f"**Pixel size**: {config_bioimageio.model_pixel_size:.3f} µm/pixel\n")
        f.write(f"**Minimum object diameter**: {config_bioimageio.min_diameter} pixels\n\n")

        # Citation
        if config_bioimageio.citation_text or config_bioimageio.citation_doi:
            f.write("## Citation\n")
            if config_bioimageio.citation_text:
                f.write(f"{config_bioimageio.citation_text}\n\n")
            if config_bioimageio.citation_doi:
                f.write(f"DOI: [{config_bioimageio.citation_doi}](https://doi.org/{config_bioimageio.citation_doi})\n\n")

        # Authors
        if config_bioimageio.author_names:
            f.write("## Authors\n")
            for author in config_bioimageio.author_names:
                f.write(f"- {author}\n")
            f.write("\n")

        # Usage note
        f.write("## Usage\n")
        f.write("Refer to the AimSegDL documentation for inference and post-processing instructions.\n")

    return readme_filename


def generate_and_save_custom_cover(
    input_image: np.ndarray,
    output_tensor: torch.Tensor,
    min_diameter: float,
    save_path: str
):
    """
    Creates a 4-panel side-by-side cover image:
    - Region 1: Input grayscale (gray colormap)
    - Region 2: Semantic prediction (argmax, viridis colormap)
    - Region 3: Fibre distance (nipy_spectral after instance segmentation)
    - Region 4: Axon distance (nipy_spectral after instance segmentation)

    The output has the same height and width as the original image.

    Args:
        input_image (np.ndarray): Shape [1, 1, H, W]
        output_tensor (torch.Tensor): Shape [1, 3, H, W]
        save_path (str): File path to save the composite cover image
    """
    assert input_image.ndim == 4 and input_image.shape[1] == 1, "Expected input shape [1, 1, H, W]"
    assert output_tensor.ndim == 4 and output_tensor.shape[1] == 3, "Expected output shape [1, 3, H, W]"

    # Extract images
    gray = input_image[0, 0]
    semantic_logits = output_tensor[0, 0]
    fibre = segment_instances_from_sdt(output_tensor[0, 1], min_diameter=min_diameter)
    axon = segment_instances_from_sdt(output_tensor[0, 2], min_diameter=min_diameter)
    axon_mapped = map_axon_labels_to_fibres(fibre, axon)

    # Ensure semantic is class index (0, 1, 2)
    semantic_classes = semantic_logits.cpu().numpy()

    # Fill holes in semantic label 2
    label_2_mask = (semantic_classes == 2)
    filled_label_2 = binary_fill_holes(label_2_mask)
    semantic_classes[(semantic_classes != 2) & filled_label_2] = 2

    # Get colormapped RGB versions (values expected in [0, N] range)
    glasbey = get_glasbey_cmap()
    gray_rgb = cm.get_cmap("gray")(gray)[..., :3]
    semantic_rgb = cm.get_cmap("viridis")(semantic_classes / 2.0)[..., :3]
    fibre_rgb = glasbey(fibre / (fibre.max() or 1))[..., :3]
    axon_rgb = glasbey(axon_mapped / (axon_mapped.max() or 1))[..., :3]

    # Stack all full-sized images into a blank canvas
    H, W = gray.shape
    panel_width = W // 4
    canvas = np.zeros((H, W, 3), dtype=np.float32)

    # Assign each panel to a quarter region
    canvas[:, 0 * panel_width:1 * panel_width] = gray_rgb[:, 0 * panel_width:1 * panel_width]
    canvas[:, 1 * panel_width:2 * panel_width] = fibre_rgb[:, 1 * panel_width:2 * panel_width]
    canvas[:, 2 * panel_width:3 * panel_width] = semantic_rgb[:, 2 * panel_width:3 * panel_width]
    canvas[:, 3 * panel_width:4 * panel_width] = axon_rgb[:, 3 * panel_width:4 * panel_width]

    # Save final image
    Image.fromarray((canvas * 255).astype(np.uint8)).save(save_path)
    print(f"Custom cover saved to {save_path}")


def export_bioimageio(config_bioimageio: BioimageioExportConfig):
    """
    Export AimSegDL TorchScript model to a BioImage.IO package.
    """
    os.makedirs(config_bioimageio.output_dir, exist_ok=True)

    # Load model and wrap in pipeline
    model = load_model(config_bioimageio.model_path, device="cpu")
    model.eval()
    wrapped_model = Pipeline(model)

    # Prepare test input
    img = np.array(Image.open(config_bioimageio.test_img_path)).astype(np.float32)
    input_ = normalize(img)[None, None]  # [1, 1, H, W]
    input_tensor = torch.from_numpy(input_)

    # Script and save TorchScript model
    scripted_model = torch.jit.script(wrapped_model)
    torch.jit.save(scripted_model, os.path.join(config_bioimageio.output_dir, "weights.pt"))

    # Save test input and output
    np.save(os.path.join(config_bioimageio.output_dir, "test-input.npy"), input_tensor.detach().cpu().numpy())
    with torch.no_grad():
        output = scripted_model(input_tensor)

    # Ensure it's raw NumPy, not xarray or torch tensor
    output_np = output.detach().cpu().numpy().astype(np.float32)

    # Avoid fancy indexing issues — make sure it's a plain ndarray
    assert isinstance(output_np, np.ndarray) and output_np.ndim == 4

    np.save(os.path.join(config_bioimageio.output_dir, "test-output.npy"), output_np)

    # Confirm similarity
    expected = output.cpu().numpy()
    predicted = np.load(os.path.join(config_bioimageio.output_dir, "test-output.npy"))

    np.testing.assert_allclose(expected, predicted, rtol=1e-3, atol=1e-3)
    assert input_tensor.shape[2:] == output.shape[2:], "Input/output shape mismatch!"


    # Write README inside output_dir
    readme_filename = write_readme(config_bioimageio)

    # Define input
    input_descr = InputTensorDescr(
        id=TensorId("raw"),
        axes=[
            BatchAxis(),
            ChannelAxis(id=AxisId("channel"), channel_names=[Identifier("gray")]),
            SpaceInputAxis(id=AxisId("y"), size=ParameterizedSize(min=512, step=1), scale=config_bioimageio.model_pixel_size, unit="micrometer"),
            SpaceInputAxis(id=AxisId("x"), size=ParameterizedSize(min=512, step=1), scale=config_bioimageio.model_pixel_size, unit="micrometer"),
        ],
        data=IntervalOrRatioDataDescr(type="float32"),
        test_tensor=FileDescr(source=os.path.join(config_bioimageio.output_dir, "test-input.npy"))
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
            SpaceOutputAxis(id=AxisId("y"), size=SizeReference(tensor_id=TensorId("raw"), axis_id=AxisId("y")), scale=config_bioimageio.model_pixel_size, unit="micrometer"),
            SpaceOutputAxis(id=AxisId("x"), size=SizeReference(tensor_id=TensorId("raw"), axis_id=AxisId("x")), scale=config_bioimageio.model_pixel_size, unit="micrometer"),
        ],
        test_tensor=FileDescr(source=os.path.join(config_bioimageio.output_dir, "test-output.npy"))
    )


    generate_and_save_custom_cover(
        input_image=input_,
        output_tensor=output,
        min_diameter=config_bioimageio.min_diameter,
        save_path=os.path.join(config_bioimageio.output_dir, "cover.png")
    )


    #Define model
    model_descr = ModelDescr(
        name=config_bioimageio.model_name,
        version=config_bioimageio.model_version,
        description="AimSegDL TorchScript model for axon/fibre segmentation.",
        authors=[Author(name=name) for name in config_bioimageio.author_names],
        license=LicenseId(config_bioimageio.license_id),
        documentation=RelativeFilePath(Path(os.path.relpath(os.path.join(config_bioimageio.output_dir, readme_filename)))),
        covers=[os.path.join(config_bioimageio.output_dir, "cover.png")],
        git_repo=HttpUrl("https://github.com/paucabar/aimseg-dl"),
        inputs=[input_descr],
        outputs=[output_descr],
        weights=WeightsDescr(
            torchscript=TorchscriptWeightsDescr(
                source=RelativeFilePath(Path(os.path.relpath(os.path.join(config_bioimageio.output_dir, "weights.pt")))),
                pytorch_version=torch.__version__,
            )
        ),
        cite=[CiteEntry(text=config_bioimageio.citation_text, doi=Doi(config_bioimageio.citation_doi))],  # TODO: update DOI
        config={
            "pixel_size": config_bioimageio.model_pixel_size, # custom field
            "min_diameter": config_bioimageio.min_diameter, # custom field
            "predict_inner_tongue": config_bioimageio.predict_inner_tongue} # custom field
    )

    # Save model package
    zip_path = Path(config_bioimageio.output_dir) / f"{config_bioimageio.model_name}-{config_bioimageio.model_version}.zip"
    package_path = save_bioimageio_package(model_descr, output_path=zip_path)
    print("Saved model package:", package_path)

    # Validate model
    if config_bioimageio.validate:
        summary = test_model(model_descr, weight_format="torchscript", test_tolerance=1e-2)
        summary.display()
