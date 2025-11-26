import argparse
import json
import os
import time

import requests
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential


def get_access_token() -> str:
    """Get an access token for Azure ML using managed identity or default credentials."""
    credentials_to_try = [
        ("ManagedIdentityCredential", lambda: ManagedIdentityCredential()),
        ("DefaultAzureCredential", lambda: DefaultAzureCredential()),
    ]

    for cred_name, cred_factory in credentials_to_try:
        try:
            print(f"Trying {cred_name}...")
            credential = cred_factory()
            token = credential.get_token("https://management.azure.com/.default")
            print(f"✅ Successfully obtained token using {cred_name}")
            return token.token
        except Exception as e:
            print(f"⚠️ {cred_name} failed: {str(e)[:100]}")
            continue

    raise RuntimeError("Failed to obtain access token with any credential type")


def register_model_via_rest(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    model_name: str,
    model_version: str,
    model_path: str,
    model_type: str,
    description: str,
    access_token: str,
) -> dict:
    """Register a model using the Azure ML REST API directly."""

    # Azure ML REST API endpoint for creating model versions
    base_url = "https://management.azure.com"
    api_version = "2023-04-01"

    url = (
        f"{base_url}/subscriptions/{subscription_id}"
        f"/resourceGroups/{resource_group}"
        f"/providers/Microsoft.MachineLearningServices"
        f"/workspaces/{workspace_name}"
        f"/models/{model_name}"
        f"/versions/{model_version}"
        f"?api-version={api_version}"
    )

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    # Map model type to Azure ML asset type
    asset_type_map = {
        "custom_model": "CustomModel",
        "mlflow_model": "MLFlowModel",
        "triton_model": "TritonModel",
    }
    model_asset_type = asset_type_map.get(model_type, "CustomModel")

    # Request body for model registration
    body = {
        "properties": {
            "description": description,
            "modelType": model_asset_type,
            "modelUri": f"azureml://jobs/{os.environ.get('AZUREML_RUN_ID', 'unknown')}/outputs/artifacts/paths/{os.path.basename(model_path)}/",
            "isAnonymous": False,
            "tags": {
                "framework": "tensorflow",
                "task": "image-classification",
                "dataset": "sports-balls",
                "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "properties": {
                "registered_from": "pipeline",
                "run_id": os.environ.get("AZUREML_RUN_ID", "unknown"),
            },
        }
    }

    print(f"Registering model via REST API...")
    print(f"URL: {url}")
    print(f"Model URI: {body['properties']['modelUri']}")

    response = requests.put(url, headers=headers, json=body)

    if response.status_code in [200, 201]:
        print(f"✅ Model registered successfully (HTTP {response.status_code})")
        return response.json()
    else:
        print(f"❌ Failed to register model (HTTP {response.status_code})")
        print(f"Response: {response.text}")

        # Try alternative: use datastore path
        print("\nRetrying with file:// path...")
        body["properties"]["modelUri"] = f"file://{model_path}"

        response = requests.put(url, headers=headers, json=body)

        if response.status_code in [200, 201]:
            print(
                f"✅ Model registered successfully on retry (HTTP {response.status_code})"
            )
            return response.json()
        else:
            raise RuntimeError(
                f"Failed to register model: HTTP {response.status_code} - {response.text}"
            )


def upload_and_register_model(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    model_name: str,
    model_version: str,
    model_path: str,
    model_type: str,
    description: str,
    access_token: str,
) -> dict:
    """
    Alternative approach: Create model container first, then version.
    This avoids the read permission issue.
    """
    base_url = "https://management.azure.com"
    api_version = "2023-04-01"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    # Step 1: Ensure model container exists (create if not)
    container_url = (
        f"{base_url}/subscriptions/{subscription_id}"
        f"/resourceGroups/{resource_group}"
        f"/providers/Microsoft.MachineLearningServices"
        f"/workspaces/{workspace_name}"
        f"/models/{model_name}"
        f"?api-version={api_version}"
    )

    container_body = {
        "properties": {
            "description": f"Model container for {model_name}",
        }
    }

    print(f"Creating/updating model container: {model_name}")
    response = requests.put(container_url, headers=headers, json=container_body)

    if response.status_code not in [200, 201]:
        print(f"⚠️ Model container creation returned: HTTP {response.status_code}")
        print(f"   This may be okay if container already exists.")
    else:
        print(f"✅ Model container ready")

    # Step 2: Create model version
    version_url = (
        f"{base_url}/subscriptions/{subscription_id}"
        f"/resourceGroups/{resource_group}"
        f"/providers/Microsoft.MachineLearningServices"
        f"/workspaces/{workspace_name}"
        f"/models/{model_name}"
        f"/versions/{model_version}"
        f"?api-version={api_version}"
    )

    # Get the run ID for constructing the artifact path
    run_id = os.environ.get("AZUREML_RUN_ID", "")

    # Construct the model URI - this points to the output artifacts
    # Format: azureml://jobs/<run_id>/outputs/<output_name>
    model_uri = f"azureml://jobs/{run_id}/outputs/artifacts/paths/output_model/"

    # Alternative formats to try if the first one doesn't work
    model_uri_alternatives = [
        model_uri,
        f"azureml://datastores/workspaceartifactstore/paths/ExperimentRun/dcid.{run_id}/output_model/",
        f"runs:/{run_id}/output_model",
        model_path,  # Direct local path as fallback
    ]

    asset_type_map = {
        "custom_model": "CustomModel",
        "mlflow_model": "MLFlowModel",
        "triton_model": "TritonModel",
    }
    model_asset_type = asset_type_map.get(model_type, "CustomModel")

    for uri in model_uri_alternatives:
        version_body = {
            "properties": {
                "description": description,
                "modelType": model_asset_type,
                "modelUri": uri,
                "isAnonymous": False,
                "tags": {
                    "framework": "tensorflow",
                    "task": "image-classification",
                    "dataset": "sports-balls",
                },
                "properties": {
                    "run_id": run_id,
                    "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            }
        }

        print(f"\nCreating model version {model_version} with URI: {uri}")
        response = requests.put(version_url, headers=headers, json=version_body)

        if response.status_code in [200, 201]:
            print(f"✅ Model version created successfully!")
            return response.json()
        else:
            print(f"⚠️ Failed with URI '{uri}': HTTP {response.status_code}")
            if response.text:
                try:
                    error_detail = json.loads(response.text)
                    print(
                        f"   Error: {error_detail.get('error', {}).get('message', response.text)[:200]}"
                    )
                except:
                    print(f"   Response: {response.text[:200]}")

    # If all alternatives failed, raise an error
    raise RuntimeError(f"Failed to register model after trying all URI formats")


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
    print("Model Registration (REST API)")
    print("=" * 60)
    print(f"Model name: {args.model_name}")
    print(f"Model path: {args.model_path}")
    print(f"Model type: {args.model_type}")
    print()

    # List files in model path
    if os.path.exists(args.model_path):
        print(f"Files in model path:")
        for root, dirs, files in os.walk(args.model_path):
            for f in files:
                filepath = os.path.join(root, f)
                size = os.path.getsize(filepath)
                print(f"  {filepath} ({size} bytes)")
    else:
        print(f"⚠️ Model path does not exist: {args.model_path}")
    print()

    # Get Azure ML workspace details from environment variables
    subscription_id = os.environ.get("AZUREML_ARM_SUBSCRIPTION")
    resource_group = os.environ.get("AZUREML_ARM_RESOURCEGROUP")
    workspace_name = os.environ.get("AZUREML_ARM_WORKSPACE_NAME")
    run_id = os.environ.get("AZUREML_RUN_ID", "unknown")

    if not all([subscription_id, resource_group, workspace_name]):
        raise ValueError(
            "Missing required environment variables: "
            "AZUREML_ARM_SUBSCRIPTION, AZUREML_ARM_RESOURCEGROUP, AZUREML_ARM_WORKSPACE_NAME"
        )

    print(f"Workspace: {workspace_name}")
    print(f"Resource group: {resource_group}")
    print(f"Subscription: {subscription_id}")
    print(f"Run ID: {run_id}")
    print()

    # Get access token
    access_token = get_access_token()

    # Generate a timestamp-based version
    timestamp_version = str(int(time.time()))
    print(f"Model version: {timestamp_version}")
    print()

    description = f"Sports Ball Classification CNN model - registered at {time.strftime('%Y-%m-%d %H:%M:%S')}"

    # Register the model using REST API
    result = upload_and_register_model(
        subscription_id=subscription_id,
        resource_group=resource_group,
        workspace_name=workspace_name,
        model_name=args.model_name,
        model_version=timestamp_version,
        model_path=args.model_path,
        model_type=args.model_type,
        description=description,
        access_token=access_token,
    )

    # Extract model info from result
    model_id = result.get("id", f"azureml:{args.model_name}:{timestamp_version}")
    model_version = result.get("name", timestamp_version)

    print()
    print("✅ Model registered successfully!")
    print(f"   Name: {args.model_name}")
    print(f"   Version: {model_version}")
    print(f"   ID: {model_id}")

    # Save registration details to output folder
    os.makedirs(args.registration_details, exist_ok=True)

    registration_info = {
        "name": args.model_name,
        "version": str(model_version),
        "id": model_id,
        "type": args.model_type,
        "run_id": run_id,
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
