from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class BioimageioExportConfig:
    model_path: str                    # Path to .pth or .pt file
    model_name: str                    # Used as name of model package
    test_img_path: str                 # Input image for generating test input/output
    output_dir: str = "bioimageio_model"  # Exported model directory
    model_pixel_size: float               # µm per pixel
    readme_text: Optional[str] = None      # Optional custom README text
    cover_image: Optional[str] = None      # Optional manual cover path override
    citation_text: Optional[str] = None    # Optional citation string
    citation_doi: Optional[str] = None     # Optional DOI string
    author_names: List[str] = field(default_factory=lambda: ["Your Name"])  # Multiple authors supported
    license_id: str = "CC-BY-4.0"          # BioImage.IO License ID
