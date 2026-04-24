import argparse
from axonpath.export_utils.create_bioimageio_model import export_bioimageio
from axonpath.export_utils.config import BioimageioExportConfig

def main():
    parser = argparse.ArgumentParser(description="Export trained axonpath model to BioImage.IO format")

    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth weights file")
    parser.add_argument("--test_img_path", type=str, required=True, help="Path to sample input image")
    parser.add_argument("--output_dir", type=str, default="bioimageio_model", help="Output directory for BioImage.IO package")
    parser.add_argument("--model_name", type=str, default="axonpath", help="Name of the model")
    parser.add_argument("--model_version", type=str, default="0.1.0", help="Version of the model")
    parser.add_argument("--pixel_size", type=float, default=0.008, help="Pixel size in microns")
    parser.add_argument("--min_diameter", type=float, default=30.0, help="Expected minimum diameter")
    parser.add_argument("--citation_text", type=str, default="Carrillo-Barberà et al., 2025", help="Citation text")
    parser.add_argument("--citation_doi", type=str, default="10.1234/fake-doi-placeholder", help="Citation DOI")
    parser.add_argument("--license_id", type=str, default="CC-BY-4.0", help="License ID")
    parser.add_argument(
        "--author_names",
        nargs="+",
        default=[
            "Pau Carrillo-Barberà",
            "Thibaut Goldsborough",
            "Alan O'Callaghan",
            "Andrea Poveda Sabuco",
            "Chiara Sgattoni",
            "Jose Antonio Gómez Sánchez",
            "Ana Rondelli",
            "Anna Williams",
            "Peter Bankhead"
        ],
        help="List of author names"
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=512,
        help="Minimum input tile size enforced by the Pipeline wrapper (default: 512)"
    )
    parser.add_argument(
        "--predict_inner_cylinder",
        action="store_true",
        default=False,
        help="If passed, marks the model as predicting inner tongue (cylinder) in addition to fibre and axon"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="If passed, validates the exported BioImage.IO model"
    )


    args = parser.parse_args()

    config = BioimageioExportConfig(
        model_path=args.model_path,
        model_name=args.model_name,
        model_version=args.model_version,
        test_img_path=args.test_img_path,
        output_dir=args.output_dir,
        model_pixel_size=args.pixel_size,
        min_diameter=args.min_diameter,
        tile_size=args.tile_size,
        predict_inner_cylinder=args.predict_inner_cylinder,
        citation_text=args.citation_text,
        citation_doi=args.citation_doi,
        author_names=args.author_names,
        license_id=args.license_id,
        validate=args.validate,
    )

    export_bioimageio(config)


if __name__ == "__main__":
    main()
