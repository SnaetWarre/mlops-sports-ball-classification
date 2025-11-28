import argparse
import json
import os
import sys
import time

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential


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
    print("Model Registration (Azure ML SDK)")
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

    # Authenticate using managed identity or default credentials
    print("Authenticating with Azure...")
    try:
        # Try ManagedIdentityCredential first (for compute clusters)
        credential = ManagedIdentityCredential()
        # Test the credential
        credential.get_token("https://management.azure.com/.default")
        print("Using Managed Identity authentication")
    except Exception as e:
        print(f"Managed Identity not available: {e}")
        print("Falling back to DefaultAzureCredential...")
        credential = DefaultAzureCredential()
    print()

    # Create ML Client
    print("Creating Azure ML client...")
    try:
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )
        print("Azure ML client created successfully")
    except Exception as e:
        print(f"ERROR: Failed to create ML client: {e}")
        sys.exit(1)
    print()

    # Create model description
    description = (
        f"Sports Ball Classification CNN model - "
        f"Run ID: {run_id} - "
        f"Registered: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Create Model object
    print("Creating model object...")
    model = Model(
        name=args.model_name,
        version=timestamp_version,
        path=args.model_path,
        type=args.model_type,
        description=description,
        tags={
            "run_id": run_id,
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "framework": "tensorflow",
        },
    )
    print()

    # Register the model
    print("Registering model...")
    try:
        registered_model = ml_client.models.create_or_update(model)
        print("Model registered successfully!")
    except Exception as e:
        print(f"ERROR: Model registration failed: {e}")

        # Retry without explicit version
        print()
        print("Retrying without explicit version...")
        try:
            model_retry = Model(
                name=args.model_name,
                path=args.model_path,
                type=args.model_type,
                description=description,
                tags={
                    "run_id": run_id,
                    "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "framework": "tensorflow",
                },
            )
            registered_model = ml_client.models.create_or_update(model_retry)
            print("Model registered successfully on retry!")
        except Exception as retry_error:
            print(f"ERROR: Retry also failed: {retry_error}")
            sys.exit(1)

    print()
    print("=" * 60)
    print("Registration Complete!")
    print("=" * 60)
    print(f"Model name: {registered_model.name}")
    print(f"Model version: {registered_model.version}")
    print(f"Model ID: {registered_model.id}")
    print()

    # Save registration details to output folder
    os.makedirs(args.registration_details, exist_ok=True)

    registration_info = {
        "name": registered_model.name,
        "version": str(registered_model.version),
        "id": registered_model.id,
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
