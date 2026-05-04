import argparse
import os

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: huggingface_hub. Install it with 'python3 -m pip install huggingface_hub' "
        "or 'python3 -m pip install -r requirements.txt'."
    ) from exc


def main():
    parser = argparse.ArgumentParser(description="Download a Hugging Face model snapshot into a local directory.")
    parser.add_argument("model_id", help="Model ID on Hugging Face, for example Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.expanduser("~"), "models"),
        help="Base directory where the model snapshot should be stored.",
    )
    parser.add_argument(
        "--local_name",
        default=None,
        help="Optional local directory name. Defaults to the model ID basename.",
    )
    args = parser.parse_args()

    local_name = args.local_name or args.model_id.rstrip("/").split("/")[-1]
    target_dir = os.path.join(args.output_dir, local_name)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Downloading {args.model_id} to {target_dir}")
    snapshot_download(repo_id=args.model_id, local_dir=target_dir, local_dir_use_symlinks=False)
    print(f"Done: {target_dir}")


if __name__ == "__main__":
    main()