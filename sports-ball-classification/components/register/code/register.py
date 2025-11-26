import argparse
import json
import os
import time

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential


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

    print(f"Registering model: {args.model_name}")
    print(f"Model path: {args.model_path}")
    print(f"Model type: {args.model_type}")

    # Get Azure ML workspace details from environment variables
    subscription_id = os.environ.get("AZUREML_ARM_SUBSCRIPTION")
    resource_group = os.environ.get("AZUREML_ARM_RESOURCEGROUP")
    workspace_name = os.environ.get("AZUREML_ARM_WORKSPACE_NAME")

    print(f"Connecting to workspace: {workspace_name}")
    print(f"Resource group: {resource_group}")
    print(f"Subscription: {subscription_id}")

    # Create ML client using default credentials
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    # Map model type string to AssetTypes
    model_type_map = {
        "custom_model": AssetTypes.CUSTOM_MODEL,
        "mlflow_model": AssetTypes.MLFLOW_MODEL,
        "triton_model": AssetTypes.TRITON_MODEL,
    }

    asset_type = model_type_map.get(args.model_type, AssetTypes.CUSTOM_MODEL)

    # Generate a timestamp-based version to avoid conflicts
    timestamp_version = str(int(time.time()))

    print(f"Creating model with version: {timestamp_version}")

    # Create model entity with explicit version
    model = Model(
        path=args.model_path,
        name=args.model_name,
        version=timestamp_version,
        type=asset_type,
        description=f"Sports Ball Classification model registered at {time.strftime('%Y-%m-%d %H:%M:%S')}",
    )

    print("Registering model...")

    try:
        registered_model = ml_client.models.create_or_update(model)
        print("Model registered successfully!")

    except HttpResponseError as e:
        error_message = str(e)
        if "AuthorizationFailed" in error_message and "read" in error_message.lower():
            print("ERROR: The compute identity lacks permission to read models.")
            print("This is required by the Azure ML SDK's create_or_update method.")
            print("\nTo fix this, grant the compute identity one of these roles:")
            print("  - Azure Machine Learning Data Scientist")
            print("  - AzureML Data Scientist")
            print(
                "  - A custom role with 'Microsoft.MachineLearningServices/workspaces/models/*/read' permission"
            )
            raise
        else:
            print(f"ERROR: Failed to register model: {error_message}")
            raise

    print(f"Model name: {registered_model.name}")
    print(f"Model version: {registered_model.version}")
    print(f"Model ID: {registered_model.id}")

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

    print(f"Registration details saved to: {output_file}")


if __name__ == "__main__":
    main()
