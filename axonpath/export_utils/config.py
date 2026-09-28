from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BioimageioExportConfig:
    model_path: str                    # Path to .pth or .pt file
    model_name: str                    # Used as name of model package
    model_version: str                 # Current version of the model
    test_img_path: str                 # Input image for generating test input/output
    output_dir: str                      # Exported model directory
    model_pixel_size: float               # µm per pixel
    min_diameter: float                    # Minimum fibre diameter in pixels, at model_pixel_size
    tile_size: int = 512                   # Minimum input tile size enforced by the Pipeline wrapper
    predict_inner_cylinder: bool = True      # Only predicts fibre and axon if false
    readme_text: Optional[str] = None      # Optional custom README text
    cover_image: Optional[str] = None      # Optional manual cover path override
    citation_text: Optional[str] = None    # Optional citation string
    citation_doi: Optional[str] = None     # Optional DOI string
    author_names: List[str] = field(default_factory=lambda: ["Your Name"])  # Multiple authors supported
    license_id: str = "CC-BY-4.0"          # BioImage.IO License ID
    validate: bool = False                 # Run BioimageIO test
    description: str = "AxonPath model for segmenting myelinated fibres and axons."  # One-line, model-specific description
    git_repo: str = "https://github.com/paucabar/axonpath"                            # Training code repository
    qupath_extension_url: str = "https://github.com/paucabar/qupath-extension-axonpath"  # Where to run the model
    tags: List[str] = field(default_factory=list)  # BioImage.IO search tags
