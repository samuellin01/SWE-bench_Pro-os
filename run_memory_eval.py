#!/usr/bin/env python3
"""
run_memory_eval.py - Automates the full evaluation pipeline for memory_experiments.

This script:
1. Clones/pulls the samuellin01/memory_experiments repo (or uses a provided local path)
2. Scans the sbp/ directory to find all instance_*/config/ folders with patch.diff files
3. Converts patches into the JSON format expected by swe_bench_pro_eval.py:
   [
     {
       "instance_id": "<instance_id>",
       "patch": "<patch content>",
       "prefix": "<config_name>"
     },
     ...
   ]
4. Runs swe_bench_pro_eval.py with the converted patches

The memory_experiments repo is expected to have the following structure:
  sbp/
    instance_<instance_id>/
      <config_name>/        # e.g. "no_compression"
        patch.diff          # the generated patch
        metadata.json       # run metadata (instance_id, config, success, etc.)
        agent_output.log
        token_usage.json
        traj_*.json

Usage:
  # Clone memory_experiments automatically and evaluate all configs:
  python run_memory_eval.py \\
      --raw_sample_path swe_bench_pro_full.csv \\
      --output_dir output/ \\
      --dockerhub_username myuser \\
      --scripts_dir run_scripts \\
      --use_podman

  # Use a local copy of memory_experiments and filter to one config:
  python run_memory_eval.py \\
      --memory_experiments_path /path/to/memory_experiments \\
      --config no_compression \\
      --raw_sample_path swe_bench_pro_full.csv \\
      --output_dir output/ \\
      --dockerhub_username myuser \\
      --scripts_dir run_scripts \\
      --use_podman

  # Only convert patches to JSON without running evaluation:
  python run_memory_eval.py \\
      --memory_experiments_path /path/to/memory_experiments \\
      --config no_compression \\
      --raw_sample_path swe_bench_pro_full.csv \\
      --output_dir output/ \\
      --dockerhub_username myuser \\
      --scripts_dir run_scripts \\
      --convert_only
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

MEMORY_EXPERIMENTS_REPO = "https://github.com/samuellin01/memory_experiments.git"


def clone_or_pull_repo(repo_url: str, local_path: str) -> str:
    """Clone the repo if it doesn't exist, or pull latest changes if it does.

    Args:
        repo_url: URL of the git repository to clone
        local_path: Local directory path for the repo

    Returns:
        The local_path, for convenience
    """
    if os.path.isdir(os.path.join(local_path, ".git")):
        print(f"Pulling latest changes in {local_path} ...")
        result = subprocess.run(
            ["git", "-C", local_path, "pull"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Warning: git pull failed: {result.stderr.strip()}")
        else:
            print(result.stdout.strip())
    else:
        print(f"Cloning {repo_url} into {local_path} ...")
        result = subprocess.run(
            ["git", "clone", repo_url, local_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to clone {repo_url}: {result.stderr.strip()}"
            )
        print(result.stdout.strip())
    return local_path


def gather_patches(
    sbp_dir: str,
    config_filter: Optional[str] = None,
) -> Dict[str, List[Dict[str, str]]]:
    """Scan sbp/ directory and gather patches grouped by config name.

    Args:
        sbp_dir: Path to the sbp/ directory inside memory_experiments
        config_filter: If provided, only include patches for this config name.
                       Use None to include all configs.

    Returns:
        Dict mapping config_name -> list of patch dicts suitable for
        swe_bench_pro_eval.py:
          [{"instance_id": ..., "patch": ..., "prefix": ...}, ...]
    """
    sbp_path = Path(sbp_dir)
    if not sbp_path.exists():
        raise FileNotFoundError(f"sbp/ directory not found: {sbp_dir}")

    patches_by_config: Dict[str, List[Dict[str, str]]] = {}

    for instance_dir in sorted(sbp_path.iterdir()):
        if not instance_dir.is_dir():
            continue
        if not instance_dir.name.startswith("instance_"):
            continue

        # Default instance_id is the folder name; may be overridden by metadata
        folder_instance_id = instance_dir.name

        for config_dir in sorted(instance_dir.iterdir()):
            if not config_dir.is_dir():
                continue

            config_name = config_dir.name

            if config_filter is not None and config_name != config_filter:
                continue

            patch_file = config_dir / "patch.diff"
            if not patch_file.exists():
                print(f"Warning: No patch.diff found in {config_dir}")
                continue

            # Prefer instance_id from metadata.json if available
            instance_id = folder_instance_id
            metadata_file = config_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file) as fh:
                        metadata = json.load(fh)
                    if "instance_id" in metadata:
                        instance_id = metadata["instance_id"]
                except Exception as exc:
                    print(
                        f"Warning: Could not read {metadata_file}: {exc}"
                    )

            try:
                patch_content = patch_file.read_text()
            except Exception as exc:
                print(f"Error reading {patch_file}: {exc}")
                continue

            patches_by_config.setdefault(config_name, []).append(
                {
                    "instance_id": instance_id,
                    "patch": patch_content,
                    "prefix": config_name,
                }
            )

    return patches_by_config


def build_eval_command(
    eval_script: str,
    raw_sample_path: str,
    patches_path: str,
    output_dir: str,
    dockerhub_username: str,
    scripts_dir: str,
    num_workers: int,
    use_local_docker: bool = False,
    use_podman: bool = False,
    docker_platform: Optional[str] = None,
    redo: bool = False,
    block_network: bool = False,
) -> List[str]:
    """Build the command list for invoking swe_bench_pro_eval.py."""
    cmd = [
        sys.executable,
        eval_script,
        "--raw_sample_path", raw_sample_path,
        "--patch_path", patches_path,
        "--output_dir", output_dir,
        "--dockerhub_username", dockerhub_username,
        "--scripts_dir", scripts_dir,
        "--num_workers", str(num_workers),
    ]
    if use_local_docker:
        cmd.append("--use_local_docker")
    if use_podman:
        cmd.append("--use_podman")
    if docker_platform:
        cmd.extend(["--docker_platform", docker_platform])
    if redo:
        cmd.append("--redo")
    if block_network:
        cmd.append("--block_network")
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automate the memory_experiments evaluation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- memory_experiments source ---
    parser.add_argument(
        "--memory_experiments_path",
        type=str,
        default=None,
        help=(
            "Path to a local memory_experiments repo. "
            "If the path does not exist, the repo is cloned there. "
            "Defaults to /tmp/memory_experiments when omitted."
        ),
    )

    # --- patch selection ---
    parser.add_argument(
        "--config",
        type=str,
        default="all",
        help=(
            "Config subdirectory name to evaluate (e.g. 'no_compression'). "
            "Use 'all' (default) to evaluate every config found."
        ),
    )

    # --- optional patch output path override ---
    parser.add_argument(
        "--patches_output",
        type=str,
        default=None,
        help=(
            "Path to write the converted patches JSON file. "
            "Defaults to <output_dir>/patches_<config>.json for each config. "
            "When evaluating multiple configs this flag is ignored."
        ),
    )

    # --- control flags ---
    parser.add_argument(
        "--convert_only",
        action="store_true",
        help="Only convert patches to JSON; do not run swe_bench_pro_eval.py.",
    )

    # --- swe_bench_pro_eval.py arguments ---
    parser.add_argument(
        "--raw_sample_path",
        required=True,
        help="Path to the raw sample CSV or JSONL file (passed to swe_bench_pro_eval.py).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to store evaluation outputs.",
    )
    parser.add_argument(
        "--dockerhub_username",
        required=True,
        help="Docker Hub username (passed to swe_bench_pro_eval.py).",
    )
    parser.add_argument(
        "--scripts_dir",
        required=True,
        help="Directory containing local run scripts (passed to swe_bench_pro_eval.py).",
    )
    parser.add_argument(
        "--use_local_docker",
        action="store_true",
        help="Run locally with Docker instead of Modal.",
    )
    parser.add_argument(
        "--use_podman",
        action="store_true",
        help="Run locally with Podman instead of Modal or Docker.",
    )
    parser.add_argument(
        "--docker_platform",
        default=None,
        help="Container platform override (e.g. linux/amd64).",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Redo evaluations even if output already exists.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=50,
        help="Number of parallel workers for swe_bench_pro_eval.py (default: 50).",
    )
    parser.add_argument(
        "--block_network",
        action="store_true",
        help="Block network access inside the evaluation container.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Step 1: Resolve memory_experiments location
    # ------------------------------------------------------------------
    mem_path = args.memory_experiments_path or "/tmp/memory_experiments"

    clone_or_pull_repo(MEMORY_EXPERIMENTS_REPO, mem_path)

    sbp_dir = os.path.join(mem_path, "sbp")

    # ------------------------------------------------------------------
    # Step 2: Scan sbp/ and convert patches
    # ------------------------------------------------------------------
    config_filter = None if args.config == "all" else args.config
    print(f"\nScanning {sbp_dir} for patches (config filter: {config_filter or 'all'}) ...")

    patches_by_config = gather_patches(sbp_dir, config_filter=config_filter)

    if not patches_by_config:
        print("No patches found. Check that the sbp/ directory exists and contains patch.diff files.")
        sys.exit(1)

    print(f"\nDiscovered {len(patches_by_config)} config(s):")
    for cfg, patches in patches_by_config.items():
        print(f"  {cfg}: {len(patches)} patch(es)")

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 3 (& 4): Write patches JSON, then optionally run eval
    # ------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_script = os.path.join(script_dir, "swe_bench_pro_eval.py")

    for config_name, patches in patches_by_config.items():
        # Determine patches JSON path
        if args.patches_output and len(patches_by_config) == 1:
            patches_path = args.patches_output
        else:
            patches_path = os.path.join(
                args.output_dir, f"patches_{config_name}.json"
            )

        with open(patches_path, "w") as fh:
            json.dump(patches, fh, indent=2)
        print(f"\nWrote {len(patches)} patches for config '{config_name}' to {patches_path}")

        if args.convert_only:
            continue

        # Per-config output subdirectory to avoid collisions
        config_output_dir = os.path.join(args.output_dir, config_name)
        os.makedirs(config_output_dir, exist_ok=True)

        cmd = build_eval_command(
            eval_script=eval_script,
            raw_sample_path=args.raw_sample_path,
            patches_path=patches_path,
            output_dir=config_output_dir,
            dockerhub_username=args.dockerhub_username,
            scripts_dir=args.scripts_dir,
            num_workers=args.num_workers,
            use_local_docker=args.use_local_docker,
            use_podman=args.use_podman,
            docker_platform=args.docker_platform,
            redo=args.redo,
            block_network=args.block_network,
        )

        print(f"\n{'='*60}")
        print(f"Running evaluation for config: {config_name}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"Warning: swe_bench_pro_eval.py exited with code {result.returncode} "
                f"for config '{config_name}'"
            )

        # Write per-instance eval scores to the memory_experiments repo
        eval_results_path = os.path.join(config_output_dir, "eval_results.json")
        if os.path.isfile(eval_results_path):
            with open(eval_results_path) as fh:
                instance_results = json.load(fh)
            overall_accuracy = (
                sum(instance_results.values()) / len(instance_results)
                if instance_results
                else 0.0
            )
            scores_payload = {
                "config": config_name,
                "overall_accuracy": overall_accuracy,
                "instance_results": instance_results,
            }
            scores_sbp_dir = os.path.join(mem_path, "sbp")
            os.makedirs(scores_sbp_dir, exist_ok=True)
            scores_path = os.path.join(scores_sbp_dir, f"eval_scores_{config_name}.json")
            with open(scores_path, "w") as fh:
                json.dump(scores_payload, fh, indent=2)
            print(f"\nWrote eval scores to {scores_path}")

            repo_path = mem_path
            git_commands = [
                ("add", ["git", "-C", repo_path, "add", scores_path]),
                (
                    "commit",
                    [
                        "git", "-C", repo_path, "commit", "-m",
                        f"Add eval scores for config '{config_name}'",
                    ],
                ),
                ("push", ["git", "-C", repo_path, "push"]),
            ]
            for verb, git_cmd in git_commands:
                git_result = subprocess.run(git_cmd, capture_output=True, text=True)
                if git_result.returncode != 0:
                    # "nothing to commit" is not an error — skip silently
                    if verb == "commit" and "nothing to commit" in git_result.stdout + git_result.stderr:
                        print("No changes to commit for eval scores (file unchanged).")
                        break
                    print(
                        f"Warning: git {verb} failed: "
                        f"{git_result.stderr.strip() or git_result.stdout.strip()}"
                    )
                else:
                    if git_result.stdout.strip():
                        print(git_result.stdout.strip())
        else:
            print(
                f"Warning: eval_results.json not found at {eval_results_path}; "
                "skipping eval scores write."
            )

    if args.convert_only:
        print("\nPatch conversion complete (--convert_only was set; evaluation skipped).")
    else:
        print("\nAll evaluations complete.")


if __name__ == "__main__":
    main()
