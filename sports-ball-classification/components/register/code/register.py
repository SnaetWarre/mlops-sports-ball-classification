import argparse
import json
import os
import subprocess
import sys
import time


def run_command(command: list[str], check: bool = True) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, and stderr."""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name for the registered model"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model folder"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="custom_model",
        help="Type of model (custom_model, mlflow_model, triton_model)",
    )
    parser.add_argument(
        "--registration_details",
        type=str,
        required=True,
        help="Output folder for registration details",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Model Registration (Azure CLI)")
    print("=" * 60)
    print(f"Model name: {args.model_name}")
    print(f"Model path: {args.model_path}")
    print(f"Model type: {args.model_type}")
    print()

    # List files in model path
    if os.path.exists(args.model_path):
        print("Files in model path:")
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(args.model_path):
            for f in files:
                filepath = os.path.join(root, f)
                size = os.path.getsize(filepath)
                total_size += size
                file_count += 1
                print(f"  {filepath} ({size:,} bytes)")
        print(f"Total: {file_count} files, {total_size:,} bytes")
    else:
        print(f"ERROR: Model path does not exist: {args.model_path}")
        sys.exit(1)
    print()

    # Get Azure ML workspace details from environment variables
    subscription_id = os.environ.get("AZUREML_ARM_SUBSCRIPTION")
    resource_group = os.environ.get("AZUREML_ARM_RESOURCEGROUP")
    workspace_name = os.environ.get("AZUREML_ARM_WORKSPACE_NAME")
    run_id = os.environ.get("AZUREML_RUN_ID", "unknown")

    if not all([subscription_id, resource_group, workspace_name]):
        print("ERROR: Missing required environment variables")
        print(f"  AZUREML_ARM_SUBSCRIPTION: {subscription_id}")
        print(f"  AZUREML_ARM_RESOURCEGROUP: {resource_group}")
        print(f"  AZUREML_ARM_WORKSPACE_NAME: {workspace_name}")
        sys.exit(1)

    print(f"Workspace: {workspace_name}")
    print(f"Resource group: {resource_group}")
    print(f"Subscription: {subscription_id}")
    print(f"Run ID: {run_id}")
    print()

    # Generate a timestamp-based version
    timestamp_version = str(int(time.time()))
    print(f"Model version: {timestamp_version}")
    print()

    # Check if az CLI is available
    exit_code, stdout, stderr = run_command(["az", "--version"])
    if exit_code != 0:
        print("ERROR: Azure CLI not available")
        print(stderr)
        sys.exit(1)
    print("Azure CLI is available")
    print()

    # Check if ml extension is installed
    exit_code, stdout, stderr = run_command(["az", "extension", "show", "--name", "ml"])
    if exit_code != 0:
        print("Installing Azure ML CLI extension...")
        exit_code, stdout, stderr = run_command(
            ["az", "extension", "add", "--name", "ml", "-y"]
        )
        if exit_code != 0:
            print(f"ERROR: Failed to install ml extension: {stderr}")
            sys.exit(1)
    print("Azure ML CLI extension is available")
    print()

    # Set defaults
    print("Setting Azure CLI defaults...")
    run_command(
        [
            "az",
            "configure",
            "--defaults",
            f"group={resource_group}",
            f"workspace={workspace_name}",
        ]
    )
    print()

    # Build the model registration command
    description = f"Sports Ball Classification CNN model - Run ID: {run_id} - Registered: {time.strftime('%Y-%m-%d %H:%M:%S')}"

    register_command = [
        "az",
        "ml",
        "model",
        "create",
        "--name",
        args.model_name,
        "--version",
        timestamp_version,
        "--path",
        args.model_path,
        "--type",
        args.model_type,
        "--description",
        description,
        "--resource-group",
        resource_group,
        "--workspace-name",
        workspace_name,
    ]

    print("Registering model...")
    exit_code, stdout, stderr = run_command(register_command)

    if exit_code != 0:
        print()
        print("ERROR: Model registration failed!")
        print(f"Exit code: {exit_code}")
        print(f"Stdout: {stdout}")
        print(f"Stderr: {stderr}")

        # Try without specifying version (let Azure ML auto-generate)
        print()
        print("Retrying without explicit version...")
        register_command_retry = [
            "az",
            "ml",
            "model",
            "create",
            "--name",
            args.model_name,
            "--path",
            args.model_path,
            "--type",
            args.model_type,
            "--description",
            description,
            "--resource-group",
            resource_group,
            "--workspace-name",
            workspace_name,
        ]

        exit_code, stdout, stderr = run_command(register_command_retry)

        if exit_code != 0:
            print()
            print("ERROR: Retry also failed!")
            print(f"Stderr: {stderr}")
            sys.exit(1)

    print()
    print("Model registered successfully!")
    print(f"Output: {stdout}")

    # Parse the output to get model details
    try:
        model_info = json.loads(stdout) if stdout.strip() else {}
        model_version = model_info.get("version", timestamp_version)
        model_id = model_info.get("id", f"azureml:{args.model_name}:{model_version}")
        model_name = model_info.get("name", args.model_name)
    except json.JSONDecodeError:
        model_version = timestamp_version
        model_id = f"azureml:{args.model_name}:{model_version}"
        model_name = args.model_name

    print()
    print("=" * 60)
    print("Registration Complete!")
    print("=" * 60)
    print(f"Model name: {model_name}")
    print(f"Model version: {model_version}")
    print(f"Model ID: {model_id}")
    print()

    # Save registration details to output folder
    os.makedirs(args.registration_details, exist_ok=True)

    registration_info = {
        "name": model_name,
        "version": str(model_version),
        "id": model_id,
        "type": args.model_type,
        "run_id": run_id,
        "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "workspace": workspace_name,
        "resource_group": resource_group,
    }

    output_file = os.path.join(args.registration_details, "registration_details.json")
    with open(output_file, "w") as f:
        json.dump(registration_info, f, indent=2)

    print(f"Registration details saved to: {output_file}")
    print(json.dumps(registration_info, indent=2))


if __name__ == "__main__":
    main()
