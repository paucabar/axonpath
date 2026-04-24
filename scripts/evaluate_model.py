import argparse
import torch
from axonpath.evaluation.evaluate_model import (
    evaluate_model_on_testset,
    compute_imagewise_means,
    plot_summary_bar
)

def main():
    parser = argparse.ArgumentParser(description="Evaluate axonpath model on test set")

    parser.add_argument("--test_dir", type=str, required=True, help="Path to test tiles directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (.pth file)")
    parser.add_argument("--min_diameter", type=float, default=30.0, help="Minimum fibre diameter in µm")
    parser.add_argument("--output_csv", type=str, default="Evaluation_Results", help="Prefix for output CSV files")
    parser.add_argument("--display_figure", action="store_true", help="Show ground truth and target tiles")
    parser.add_argument("--no_plots", action="store_true", help="Disable summary bar plots")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Evaluate the model
    fibre_results, axon_results, inner_tongue_results = evaluate_model_on_testset(
        model_path=args.model_path,
        test_dir=args.test_dir,
        device=device,
        min_diameter=args.min_diameter,
        output_csv=args.output_csv,
        display_figure=args.display_figure,
        show_plots=not args.no_plots
    )

    if not args.no_plots:
        # Summary metrics and plots
        mean_metrics_fibre = compute_imagewise_means(fibre_results)
        plot_summary_bar(mean_metrics_fibre, "Fibre")

        mean_metrics_axon = compute_imagewise_means(axon_results)
        plot_summary_bar(mean_metrics_axon, "Axon")

        mean_metrics_inner_tongue = compute_imagewise_means(inner_tongue_results)
        plot_summary_bar(mean_metrics_inner_tongue, "Inner Tongue")

if __name__ == "__main__":
    main()
