"""
The script is used to evaluate the performance of the SWEAP Pro agent with Modal,
local Docker, or local Podman.

This evaluation script:
1. Takes a CSV file containing test cases and a JSON file containing patches
2. Runs each patch in a sandbox environment using Docker Hub images
3. Executes the tests using local run scripts and collects results
4. Calculates overall accuracy based on test pass or fail status

Usage:
python swe_bench_pro_eval.py \
    --raw_sample_path=data.csv \
    --patch_path={OUTPUT}/gold_patches.json \
    --output_dir={OUTPUT}/ \
    --scripts_dir=run_scripts \
    --num_workers=100 \
    --dockerhub_username=your-username \
    --use_podman

It expects:
- Local run scripts in run_scripts/{instance_id}/run_script.sh
- Local parser scripts in run_scripts/{instance_id}/parser.py
- CSV file with columns: instance_id, before_repo_set_cmd, selected_test_files_to_run,
  base_commit, base_dockerfile, instance_dockerfile, FAIL_TO_PASS, PASS_TO_PASS

And the generated patch file (gold_patches.json) should have the following format:
[
    {
        "instance_id": "unique_id",
        "patch": "git patch content",
        "prefix": "optional_prefix"
    },
    ...
]
"""

import argparse
import concurrent.futures
import json
import os
import platform as py_platform
import subprocess

try:
    import modal  # Lazy or optional: only required when not using --use_local_docker or --use_podman
except Exception:
    modal = None
try:
    import docker  # Optional: used when --use_local_docker is set
except Exception:
    docker = None
import pandas as pd
from tqdm import tqdm

from helper_code.image_uri import get_dockerhub_image_uri


# Credit: prabhuteja12
def load_base_docker(iid):
    with open(f"dockerfiles/base_dockerfile/{iid}/Dockerfile") as fp:
        return fp.read()


def instance_docker(iid):
    with open(f"dockerfiles/instance_dockerfile/{iid}/Dockerfile") as fp:
        return fp.read()


def load_local_script(scripts_dir, instance_id, script_name):
    """Load a script file from local scripts directory."""
    script_path = os.path.join(scripts_dir, instance_id, script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    with open(script_path, "r") as f:
        return f.read()


def create_entryscript(sample):
    before_repo_set_cmd = sample["before_repo_set_cmd"].strip().split("\n")[-1]
    selected_test_files_to_run = ",".join(eval(sample["selected_test_files_to_run"]))
    base_commit = sample["base_commit"]
    base_dockerfile = load_base_docker(sample["instance_id"])
    instance_dockerfile = instance_docker(sample["instance_id"])

    # Extract ENV commands from dockerfiles
    env_cmds = []
    for dockerfile_content in [base_dockerfile, instance_dockerfile]:
        for line in dockerfile_content.split("\n"):
            line = line.strip()
            if line.startswith("ENV"):
                # Convert ENV commands to export statements
                env_cmd = line.replace("ENV", "export", 1)
                env_cmds.append(env_cmd)

    env_cmds = "\n".join(env_cmds)

    entry_script = f"""
{env_cmds}
# apply patch
cd /app
git reset --hard {base_commit}
git checkout {base_commit}
git apply -v /workspace/patch.diff
{before_repo_set_cmd}
# run test and save stdout and stderr to separate files
bash /workspace/run_script.sh {selected_test_files_to_run} > /workspace/stdout.log 2> /workspace/stderr.log
# run parsing script
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
"""
    return entry_script


def create_dockerhub_tag(uid, repo_name=""):
    """
    Convert instance_id and repo name to Docker Hub compatible tag format.
    This must match the format used in the upload script.

    Args:
        uid (str): The instance_id (for example "django__django-12345")
        repo_name (str): The repository name from ECR (for example "sweap-images/nodebb.nodebb")

    Returns:
        str: Docker Hub compatible tag (for example "nodebb-nodebb-12345")
    """
    if repo_name:
        repo_base, repo_name_only = repo_name.lower().split("/")
        hsh = uid.replace("instance_", "")
        return f"{repo_base}.{repo_name_only}-{hsh}"
    else:
        image_name = "default"

    if "__" in uid and len(uid) > 9:
        tag_part = uid[9:]
    else:
        tag_part = uid

    return f"{image_name}-{tag_part}"


def prepare_run(uid, output_dir, prefix, redo):
    uid_dir = os.path.join(output_dir, uid)
    os.makedirs(uid_dir, exist_ok=True)
    output_path = os.path.join(uid_dir, f"{prefix}_output.json")
    if not redo and os.path.exists(output_path):
        print(f"Skipping {uid} - output already exists")
        with open(output_path, "r") as f:
            return json.load(f), output_path, os.path.join(uid_dir, "workspace")
    workspace_dir = os.path.join(uid_dir, "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    return None, output_path, workspace_dir


def write_patch_snapshot(output_dir, uid, prefix, patch):
    with open(os.path.join(output_dir, uid, f"{prefix}_patch.diff"), "w") as f:
        f.write(patch)


def assemble_workspace_files(uid, scripts_dir, patch, sample):
    run_script = load_local_script(scripts_dir, uid, "run_script.sh")
    parser_script = load_local_script(scripts_dir, uid, "parser.py")
    entryscript_content = create_entryscript(sample)

    files = {
        "patch.diff": patch,
        "run_script.sh": run_script,
        "parser.py": parser_script,
        "entryscript.sh": entryscript_content,
    }
    return files, entryscript_content


def write_files_modal(sandbox, files):
    for rel_path, content in files.items():
        with sandbox.open(f"/workspace/{rel_path}", "w") as f:
            f.write(content)


def write_files_local(workspace_dir, files):
    for rel_path, content in files.items():
        dst = os.path.join(workspace_dir, rel_path)
        with open(dst, "w") as f:
            f.write(content)


def save_entryscript_copy(output_dir, uid, prefix, entryscript_content):
    with open(os.path.join(output_dir, uid, f"{prefix}_entryscript.sh"), "w") as f:
        f.write(entryscript_content if entryscript_content is not None else "")


def collect_outputs_modal(sandbox, output_dir, uid, prefix):
    # Save logs first (best effort)
    try:
        with sandbox.open("/workspace/stdout.log", "r") as f_in:
            with open(os.path.join(output_dir, uid, f"{prefix}_stdout.log"), "w") as f:
                stdout_content = f_in.read()
                f.write(stdout_content if stdout_content is not None else "")
    except FileNotFoundError:
        pass
    try:
        with sandbox.open("/workspace/stderr.log", "r") as f_in:
            with open(os.path.join(output_dir, uid, f"{prefix}_stderr.log"), "w") as f:
                stderr_content = f_in.read()
                f.write(stderr_content if stderr_content is not None else "")
    except FileNotFoundError:
        pass

    # Then try to read output.json
    try:
        with sandbox.open("/workspace/output.json", "r") as f_in:
            output = json.load(f_in)
            with open(os.path.join(output_dir, uid, f"{prefix}_output.json"), "w") as f:
                json.dump(output, f)
            return output
    except FileNotFoundError:
        print(
            f"Warning: output.json not found for {uid}. Check {prefix}_stdout.log and {prefix}_stderr.log for details"
        )
        return None


def collect_outputs_local(workspace_dir, output_dir, uid, prefix):
    def _copy_safe(src_name, dest_name):
        src_path = os.path.join(workspace_dir, src_name)
        dest_path = os.path.join(output_dir, uid, dest_name)
        try:
            with open(src_path, "r") as f_in:
                content = f_in.read()
        except FileNotFoundError:
            content = ""
        with open(dest_path, "w") as f_out:
            f_out.write(content if content is not None else "")

    _copy_safe("stdout.log", f"{prefix}_stdout.log")
    _copy_safe("stderr.log", f"{prefix}_stderr.log")

    try:
        with open(os.path.join(workspace_dir, "output.json"), "r") as f_in:
            output = json.load(f_in)
            with open(os.path.join(output_dir, uid, f"{prefix}_output.json"), "w") as f:
                json.dump(output, f)
            return output
    except FileNotFoundError:
        print(
            f"Warning: output.json not found for {uid}. Check {prefix}_stdout.log and {prefix}_stderr.log for details"
        )
        return None


def eval_with_modal(
    patch,
    sample,
    output_dir,
    dockerhub_username,
    scripts_dir,
    prefix="",
    redo=False,
    block_network=False,
    docker_platform=None,
):
    if modal is None:
        raise RuntimeError(
            "modal is not installed. Install it or run with --use_local_docker or --use_podman"
        )
    uid = sample["instance_id"]
    existing_output, output_path, workspace_dir = prepare_run(
        uid, output_dir, prefix, redo
    )
    if existing_output is not None:
        return existing_output

    sandbox = None

    print(f"Running evaluation for {uid}")
    try:
        write_patch_snapshot(output_dir, uid, prefix, patch)

        try:
            files, entryscript_content = assemble_workspace_files(
                uid, scripts_dir, patch, sample
            )
        except FileNotFoundError as e:
            print(f"Error loading scripts for {uid}: {e}")
            return None

        app = modal.App.lookup(name="swe-bench-pro-eval", create_if_missing=True)

        dockerhub_image_uri = get_dockerhub_image_uri(
            uid, dockerhub_username, sample.get("repo", "")
        )
        print(f"Using Docker Hub image: {dockerhub_image_uri}")

        image = modal.Image.from_registry(dockerhub_image_uri)

        sandbox = modal.Sandbox.create(
            image=image,
            app=app,
            timeout=60 * 60,
            cpu=(1, 4),
            memory=(5 * 1024, 30 * 1024),
            block_network=block_network,
        )

        process = sandbox.exec("mkdir", "-p", "/workspace")
        process.wait()

        write_files_modal(sandbox, files)

        process = sandbox.exec("bash", "/workspace/entryscript.sh")
        process.wait()

        if process.returncode != 0:
            print(
                f"Entryscript failed for {uid} with return code: {process.returncode}"
            )
            try:
                stderr_content = getattr(process, "stderr", None)
                if stderr_content and hasattr(stderr_content, "read"):
                    error_details = stderr_content.read()
                    if error_details:
                        print(f"Error details for {uid}:")
                        print(error_details[:1000])
            except Exception as e:
                print(f"Failed to read stderr for {uid}: {e}")

        output = collect_outputs_modal(sandbox, output_dir, uid, prefix)
        if output is None:
            return None
        save_entryscript_copy(output_dir, uid, prefix, entryscript_content)

        return output
    except Exception as e:
        print(f"Error in eval_with_modal for {uid}: {repr(e)}")
        print(f"Error type: {type(e)}")
        return None
    finally:
        if sandbox:
            try:
                sandbox.terminate()
            except Exception:
                pass


def eval_with_docker(
    patch,
    sample,
    output_dir,
    dockerhub_username,
    scripts_dir,
    prefix="",
    redo=False,
    block_network=False,
    docker_platform=None,
):
    if docker is None:
        raise RuntimeError(
            "docker SDK is not installed. Install via 'pip install docker' or run with --use_podman or without --use_local_docker"
        )
    uid = sample["instance_id"]
    existing_output, output_path, workspace_dir = prepare_run(
        uid, output_dir, prefix, redo
    )
    if existing_output is not None:
        return existing_output

    print(f"Running local-docker evaluation for {uid}")

    try:
        try:
            files, entryscript_content = assemble_workspace_files(
                uid, scripts_dir, patch, sample
            )
        except FileNotFoundError as e:
            print(f"Error loading scripts for {uid}: {e}")
            return None
        write_files_local(workspace_dir, files)
        write_patch_snapshot(output_dir, uid, prefix, patch)

        dockerhub_image_uri = get_dockerhub_image_uri(
            uid, dockerhub_username, sample.get("repo", "")
        )
        print(f"Using Docker Hub image: {dockerhub_image_uri}")

        client = docker.from_env()
        try:
            if docker_platform:
                client.images.pull(dockerhub_image_uri, platform=docker_platform)
            else:
                client.images.pull(dockerhub_image_uri)
        except Exception as pull_err:
            try:
                client.images.get(dockerhub_image_uri)
                print(f"Using locally available image: {dockerhub_image_uri}")
            except Exception:
                print(
                    f"Failed to pull or find image locally for {uid}: {pull_err}"
                )
                return None

        abs_workspace_dir = os.path.abspath(workspace_dir)
        volumes = {abs_workspace_dir: {"bind": "/workspace", "mode": "rw"}}
        run_kwargs = {
            "volumes": volumes,
            "detach": True,
            "remove": True,
            "entrypoint": "/bin/bash",
            "command": ["-c", "bash /workspace/entryscript.sh"],
        }
        if block_network:
            run_kwargs["network_mode"] = "none"
        if docker_platform:
            run_kwargs["platform"] = docker_platform

        container = client.containers.run(
            dockerhub_image_uri,
            **run_kwargs,
        )

        result = container.wait()
        status_code = result.get("StatusCode", 1) if isinstance(result, dict) else 1
        if status_code != 0:
            print(
                f"Entryscript failed for {uid} with return code: {status_code}"
            )
        output = collect_outputs_local(workspace_dir, output_dir, uid, prefix)
        if output is None:
            return None
        save_entryscript_copy(output_dir, uid, prefix, entryscript_content)

        return output
    except Exception as e:
        print(f"Error in eval_with_docker for {uid}: {repr(e)}")
        print(f"Error type: {type(e)}")
        return None

def eval_with_podman(
    patch,
    sample,
    output_dir,
    dockerhub_username,
    scripts_dir,
    prefix="",
    redo=False,
    block_network=False,
    docker_platform=None,
):
    uid = sample["instance_id"]
    existing_output, output_path, workspace_dir = prepare_run(
        uid, output_dir, prefix, redo
    )
    if existing_output is not None:
        return existing_output

    print(f"Running local-podman evaluation for {uid}")

    try:
        # Set up workspace files
        try:
            files, entryscript_content = assemble_workspace_files(
                uid, scripts_dir, patch, sample
            )
        except FileNotFoundError as e:
            print(f"Error loading scripts for {uid}: {e}")
            return None

        write_files_local(workspace_dir, files)
        write_patch_snapshot(output_dir, uid, prefix, patch)

        # Get image name, and force Podman to use Docker Hub
        dockerhub_image_uri = get_dockerhub_image_uri(
            uid, dockerhub_username, sample.get("repo", "")
        )
        if dockerhub_image_uri.startswith(("docker.io/", "ghcr.io/", "quay.io/")):
            podman_image = dockerhub_image_uri
        else:
            podman_image = f"docker.io/{dockerhub_image_uri}"

        print(f"Using Podman image: {podman_image}")

        # Pull image with ignore_chown_errors
        pull_cmd = ["podman", "pull", "--storage-opt", "ignore_chown_errors=true", podman_image]
        if docker_platform:
            pull_cmd.extend(["--platform", docker_platform])

        env = os.environ.copy()
        env.setdefault("PODMAN_IGNORE_CHOWN_ERRORS", "1")

        print("Running:", " ".join(pull_cmd))
        try:
            subprocess.run(pull_cmd, check=True, env=env)
        except subprocess.CalledProcessError as pull_err:
            print(f"Failed to pull image for {uid}: {pull_err}")
            return None

        # Run container
        abs_workspace_dir = os.path.abspath(workspace_dir)

        cmd = [
            "podman",
            "run",
            "--rm",
            "-v",
            f"{abs_workspace_dir}:/workspace:rw",
            "--entrypoint",
            "/bin/bash",
        ]

        if block_network:
            cmd.extend(["--network", "none"])

        if docker_platform:
            cmd.extend(["--platform", docker_platform])

        cmd.extend([podman_image, "-c", "bash /workspace/entryscript.sh"])

        print("Running:", " ".join(cmd))
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
        )

        if result.returncode != 0:
            print(
                f"Entryscript failed for {uid} with return code: {result.returncode}"
            )
            if result.stderr:
                print("Podman stderr (first 1000 chars):")
                print(result.stderr[:1000])

        # Collect outputs and save entryscript
        output = collect_outputs_local(workspace_dir, output_dir, uid, prefix)
        if output is None:
            return None
        save_entryscript_copy(output_dir, uid, prefix, entryscript_content)

        return output
    except Exception as e:
        print(f"Error in eval_with_podman for {uid}: {repr(e)}")
        print(f"Error type: {type(e)}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run SWEAP Pro evaluations using Modal, local Docker, or local Podman "
            "with Docker Hub images and local scripts"
        )
    )
    parser.add_argument(
        "--raw_sample_path",
        required=True,
        help="Path to the raw sample CSV or JSONL file",
    )
    parser.add_argument(
        "--patch_path",
        required=True,
        help="Path to the JSON file containing patches",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to store evaluation outputs",
    )
    parser.add_argument(
        "--dockerhub_username",
        required=True,
        help="Docker Hub username where sweap-images repository is located",
    )
    parser.add_argument(
        "--scripts_dir",
        required=True,
        help="Directory containing local run scripts (for example scripts or run_scripts)",
    )
    parser.add_argument(
        "--use_local_docker",
        action="store_true",
        help="Run locally with Docker instead of Modal",
    )
    parser.add_argument(
        "--use_podman",
        action="store_true",
        help="Run locally with Podman instead of Modal or Docker",
    )
    parser.add_argument(
        "--docker_platform",
        default=None,
        help="Container platform override, for example linux/amd64; defaults to auto detect",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Redo evaluations even if output exists",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=50,
        help="Number of workers to run evaluations in parallel",
    )
    parser.add_argument(
        "--block_network",
        action="store_true",
        help="Block network access inside container",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Support both JSONL and CSV input files
    if args.raw_sample_path.endswith(".jsonl"):
        raw_sample_df = pd.read_json(args.raw_sample_path, lines=True)
    else:
        raw_sample_df = pd.read_csv(args.raw_sample_path)

    raw_sample_df = raw_sample_df.fillna("")

    raw_sample_df = raw_sample_df.set_index("instance_id", drop=False)

    with open(args.patch_path, "r") as f:
        patches_to_run = json.load(f)

    # Load existing results so previously-evaluated instances are not re-run
    existing_results_path = os.path.join(args.output_dir, "eval_results.json")
    if not args.redo and os.path.isfile(existing_results_path):
        with open(existing_results_path, "r") as f:
            eval_results = json.load(f)
        print(f"Loaded {len(eval_results)} existing result(s) from {existing_results_path}")
    else:
        eval_results = {}

    valid_patches = []
    missing_instances = []
    skipped_instances = []
    for patch_sample in patches_to_run:
        instance_id = patch_sample["instance_id"]
        if instance_id not in raw_sample_df.index:
            missing_instances.append(instance_id)
        elif not args.redo and instance_id in eval_results:
            skipped_instances.append(instance_id)
        else:
            valid_patches.append(patch_sample)

    if missing_instances:
        print(
            f"Warning: Found {len(missing_instances)} patch instances not in raw sample data:"
        )
        for missing_id in missing_instances[:5]:
            print(f"  - {missing_id}")
        if len(missing_instances) > 5:
            print(f"  ... and {len(missing_instances) - 5} more")
        print(
            f"Proceeding with {len(valid_patches)} valid patches out of {len(patches_to_run)} total patches"
        )

    if skipped_instances:
        print(
            f"Skipping {len(skipped_instances)} already-evaluated instance(s) "
            f"(pass --redo to force re-evaluation). {len(valid_patches)} new instance(s) to evaluate."
        )

    # Select runtime
    detected_platform = None
    if (args.use_local_docker or args.use_podman) and args.docker_platform is None:
        try:
            if py_platform.machine().lower() in {"arm64", "aarch64"}:
                detected_platform = "linux/amd64"
        except Exception:
            detected_platform = None

    if args.use_podman:
        eval_fn = eval_with_podman
    elif args.use_local_docker:
        eval_fn = eval_with_docker
    else:
        eval_fn = eval_with_modal

    effective_platform = args.docker_platform or detected_platform

    # Use ThreadPoolExecutor to run evaluations in parallel
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.num_workers
    ) as executor:
        future_to_patch = {
            executor.submit(
                eval_fn,
                patch_sample.get("model_patch", patch_sample.get("patch", "")),
                raw_sample_df.loc[patch_sample["instance_id"]],
                args.output_dir,
                args.dockerhub_username,
                args.scripts_dir,
                prefix=patch_sample.get("prefix", ""),
                redo=args.redo,
                block_network=args.block_network,
                docker_platform=effective_platform
                if (args.use_local_docker or args.use_podman)
                else None,
            ): patch_sample
            for patch_sample in valid_patches
        }

        pbar = tqdm(
            concurrent.futures.as_completed(future_to_patch),
            total=len(valid_patches),
        )
        for future in pbar:
            patch_sample = future_to_patch[future]
            try:
                output = future.result()
                if output is None:
                    print(
                        f'Evaluation for {patch_sample["instance_id"]} returned None'
                    )
                    eval_results[patch_sample["instance_id"]] = False
                else:
                    instance_id = patch_sample["instance_id"]
                    if instance_id not in raw_sample_df.index:
                        print(
                            f"Warning: Instance {instance_id} not found in raw sample data, skipping"
                        )
                        eval_results[instance_id] = False
                    else:
                        raw_sample = raw_sample_df.loc[instance_id]
                        passed_tests = {
                            x["name"]
                            for x in output["tests"]
                            if x["status"] == "PASSED"
                        }
                        f2p = set(eval(raw_sample["fail_to_pass"]))
                        p2p = set(eval(raw_sample["pass_to_pass"]))
                        result = (f2p | p2p) <= passed_tests
                        eval_results[instance_id] = result

                current_accuracy = sum(eval_results.values()) / len(eval_results)
                pbar.set_description(f"Accuracy: {current_accuracy:.2%}")
            except Exception as exc:
                print(
                    f'Evaluation for {patch_sample["instance_id"]} generated an exception: {exc}'
                )
                eval_results[patch_sample["instance_id"]] = False
                current_accuracy = sum(eval_results.values()) / len(eval_results)
                pbar.set_description(f"Accuracy: {current_accuracy:.2%}")

    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f)
    print("Per-instance results:")
    for instance_id, passed in sorted(eval_results.items()):
        print(f"  {instance_id}: {str(passed).lower()}")
    if eval_results:
        print("Overall accuracy: ", sum(eval_results.values()) / len(eval_results))
    else:
        print("Overall accuracy: N/A (no results)")


if __name__ == "__main__":
    main()