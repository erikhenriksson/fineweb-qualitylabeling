import argparse
from huggingface_hub import snapshot_download
from pathlib import Path


def download_cc_dump(dump_name: str, local_dir: str = "../FINEWEB") -> str:
    """
    Download a specific Common Crawl dump from the FineWeb dataset.

    Args:
        dump_name: Name of the dump (e.g., "CC-MAIN-2024-18")
        local_dir: Local directory to store the dump

    Returns:
        str: Path to the downloaded folder
    """

    dump_dir = "sample" if dump_name in ["350BT", "100BT", "10BT"] else "data"

    folder = snapshot_download(
        "HuggingFaceFW/fineweb",
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=f"{dump_dir}/{dump_name}/*",
    )
    return folder


def main():
    parser = argparse.ArgumentParser(
        description="Download Common Crawl dumps from FineWeb dataset"
    )
    parser.add_argument("dump_name", help="Name of the dump (e.g., CC-MAIN-2024-18)")
    parser.add_argument(
        "--local-dir",
        default="../FINEWEB",
        help="Local directory to store the dump (default: ../FINEWEB)",
    )

    args = parser.parse_args()

    try:
        folder = download_cc_dump(args.dump_name, args.local_dir)
        print(f"Successfully downloaded to: {folder}")
    except Exception as e:
        print(f"Error downloading dump: {e}")
        exit(1)


if __name__ == "__main__":
    main()
