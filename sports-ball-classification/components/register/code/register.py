import argparse
import json
import os
import time

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential


def get_ml_client(
    subscription_id: str, resource_group: str, workspace_name: str
) -> MLClient:
    """Try different credential types to create an ML client."""
    credentials_to_try = [
        ("ManagedIdentityCredential", lambda: ManagedIdentityCredential()),
        ("DefaultAzureCredential", lambda: DefaultAzureCredential()),
    ]

    for cred_name, cred_factory in credentials_to_try:
        try:
            print(f"Trying {cred_name}...")
            credential = cred_factory()
            client = MLClient(
                credential=credential,
                subscription_id=subscription_id,
                resource_group_name=resource_group,
                workspace_name=workspace_name,
            )
            # Test the connection by getting workspace info
            _ = client.workspace_name
            print(f"✅ Successfully connected using {cred_name}")
            return client
        except Exception as e:
            print(f"⚠️ {cred_name} failed: {str(e)[:100]}")
            continue

    raise RuntimeError("Failed to authenticate with any credential type")


def register_model_with_retry(
    ml_client: MLClient, model: Model, max_retries: int = 3
) -> Model:
    """Register model with retry logic."""
    last_error = None

    for attempt in range(max_retries):
        try:
            print(f"Registration attempt {attempt + 1}/{max_retries}...")
            registered_model = ml_client.models.create_or_update(model)
            return registered_model
        except HttpResponseError as e:
            last_error = e
            error_message = str(e)

            if "AuthorizationFailed" in error_message:
                print(f"⚠️ Authorization error on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    print(
                        f"   Waiting {wait_time}s before retry (role assignment may be propagating)..."
                    )
                    time.sleep(wait_time)
                continue
            else:
                # For other errors, don't retry
                raise

    # If we exhausted all retries, raise the last error
    if last_error:
        raise last_error
    raise RuntimeError("Failed to register model after all retries")


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
    print("Model Registration")
    print("=" * 60)
    print(f"Model name: {args.model_name}")
    print(f"Model path: {args.model_path}")
    print(f"Model type: {args.model_type}")
    print()

    # Get Azure ML workspace details from environment variables
    subscription_id = os.environ.get("AZUREML_ARM_SUBSCRIPTION")
    resource_group = os.environ.get("AZUREML_ARM_RESOURCEGROUP")
    workspace_name = os.environ.get("AZUREML_ARM_WORKSPACE_NAME")

    if not all([subscription_id, resource_group, workspace_name]):
        raise ValueError(
            "Missing required environment variables: "
            "AZUREML_ARM_SUBSCRIPTION, AZUREML_ARM_RESOURCEGROUP, AZUREML_ARM_WORKSPACE_NAME"
        )

    print(f"Workspace: {workspace_name}")
    print(f"Resource group: {resource_group}")
    print(f"Subscription: {subscription_id}")
    print()

    # Create ML client
    ml_client = get_ml_client(subscription_id, resource_group, workspace_name)

    # Map model type string to AssetTypes
    model_type_map = {
        "custom_model": AssetTypes.CUSTOM_MODEL,
        "mlflow_model": AssetTypes.MLFLOW_MODEL,
        "triton_model": AssetTypes.TRITON_MODEL,
    }
    asset_type = model_type_map.get(args.model_type, AssetTypes.CUSTOM_MODEL)

    # Generate a timestamp-based version to avoid conflicts
    timestamp_version = str(int(time.time()))
    print(f"Model version: {timestamp_version}")

    # Create model entity
    model = Model(
        path=args.model_path,
        name=args.model_name,
        version=timestamp_version,
        type=asset_type,
        description=f"Sports Ball Classification model registered at {time.strftime('%Y-%m-%d %H:%M:%S')}",
        tags={
            "framework": "tensorflow",
            "task": "image-classification",
            "dataset": "sports-balls",
        },
    )

    print()
    print("Registering model...")

    try:
        registered_model = register_model_with_retry(ml_client, model)
        print()
        print("✅ Model registered successfully!")
        print(f"   Name: {registered_model.name}")
        print(f"   Version: {registered_model.version}")
        print(f"   ID: {registered_model.id}")

    except HttpResponseError as e:
        error_message = str(e)
        print()
        print("❌ ERROR: Failed to register model")
        print(f"   {error_message[:200]}")

        if "AuthorizationFailed" in error_message:
            print()
            print("This error indicates the compute identity lacks permissions.")
            print("The GitHub Actions workflow should have granted these permissions.")
            print(
                "If this persists, manually grant the compute identity the 'Contributor' role"
            )
            print("on the Azure ML workspace.")

        raise

    # Save registration details to output folder
    os.makedirs(args.registration_details, exist_ok=True)

    registration_info = {
        "name": registered_model.name,
        "version": str(registered_model.version),
        "id": registered_model.id,
        "type": args.model_type,
        "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    output_file = os.path.join(args.registration_details, "registration_details.json")
    with open(output_file, "w") as f:
        json.dump(registration_info, f, indent=2)

    print()
    print(f"📄 Registration details saved to: {output_file}")
    print(json.dumps(registration_info, indent=2))
    print()
    print("=" * 60)
    print("Registration Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
