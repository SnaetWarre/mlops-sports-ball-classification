#!/bin/bash

# Script to grant Azure ML Compute Cluster permissions for model registration
# This fixes the "AuthorizationFailed" error when registering models

set -e

# Configuration
RESOURCE_GROUP="mlops-clean-rg"
WORKSPACE_NAME="mlops-clean-ws"
COMPUTE_NAME="warre-cluster"

echo "=================================================="
echo "Azure ML Compute Permissions Fix"
echo "=================================================="
echo ""
echo "Resource Group: $RESOURCE_GROUP"
echo "Workspace: $WORKSPACE_NAME"
echo "Compute Cluster: $COMPUTE_NAME"
echo ""

# Check if user is logged in
echo "Checking Azure CLI authentication..."
if ! az account show &> /dev/null; then
    echo "ERROR: Not logged into Azure CLI. Please run 'az login' first."
    exit 1
fi

echo "✓ Authenticated"
echo ""

# Get the current subscription
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
echo "Using subscription: $SUBSCRIPTION_ID"
echo ""

# Check if the workspace exists
echo "Verifying workspace exists..."
if ! az ml workspace show --name "$WORKSPACE_NAME" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
    echo "ERROR: Workspace '$WORKSPACE_NAME' not found in resource group '$RESOURCE_GROUP'"
    exit 1
fi
echo "✓ Workspace found"
echo ""

# Check if the compute exists
echo "Verifying compute cluster exists..."
if ! az ml compute show --name "$COMPUTE_NAME" --workspace-name "$WORKSPACE_NAME" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
    echo "ERROR: Compute cluster '$COMPUTE_NAME' not found"
    exit 1
fi
echo "✓ Compute cluster found"
echo ""

# Get the compute cluster's managed identity
echo "Getting compute cluster's managed identity..."
COMPUTE_IDENTITY=$(az ml compute show \
    --name "$COMPUTE_NAME" \
    --workspace-name "$WORKSPACE_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query identity.principal_id -o tsv)

if [ -z "$COMPUTE_IDENTITY" ] || [ "$COMPUTE_IDENTITY" = "null" ]; then
    echo "ERROR: Compute cluster does not have a managed identity enabled"
    echo ""
    echo "To fix this:"
    echo "1. Go to Azure Portal → ML Workspace → Compute → $COMPUTE_NAME"
    echo "2. Enable System Assigned Managed Identity"
    echo "3. Or recreate the compute with managed identity enabled"
    exit 1
fi

echo "✓ Managed Identity found: $COMPUTE_IDENTITY"
echo ""

# Get the workspace resource ID
echo "Getting workspace resource ID..."
WORKSPACE_ID=$(az ml workspace show \
    --name "$WORKSPACE_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query id -o tsv)

echo "✓ Workspace ID: $WORKSPACE_ID"
echo ""

# Check if the role assignment already exists
echo "Checking existing role assignments..."
EXISTING_ROLE=$(az role assignment list \
    --assignee "$COMPUTE_IDENTITY" \
    --scope "$WORKSPACE_ID" \
    --role "AzureML Data Scientist" \
    --query "[0].roleDefinitionName" -o tsv 2>/dev/null || echo "")

if [ "$EXISTING_ROLE" = "AzureML Data Scientist" ]; then
    echo "✓ Role 'AzureML Data Scientist' already assigned!"
    echo ""
    echo "The permissions are already set correctly."
    echo "If you're still seeing errors, wait 5-10 minutes for RBAC propagation."
    exit 0
fi

echo "No existing 'AzureML Data Scientist' role found"
echo ""

# Assign the role
echo "Assigning 'AzureML Data Scientist' role to compute identity..."
echo ""
echo "This role grants permissions to:"
echo "  - Read/write models and datasets"
echo "  - Run experiments and pipelines"
echo "  - Read workspace resources"
echo ""

if az role assignment create \
    --assignee "$COMPUTE_IDENTITY" \
    --role "AzureML Data Scientist" \
    --scope "$WORKSPACE_ID" &> /dev/null; then

    echo "✓ Role assigned successfully!"
    echo ""
    echo "=================================================="
    echo "SUCCESS!"
    echo "=================================================="
    echo ""
    echo "The compute cluster now has permission to register models."
    echo ""
    echo "⚠️  IMPORTANT: RBAC changes can take 5-10 minutes to propagate."
    echo ""
    echo "Next steps:"
    echo "1. Wait 5-10 minutes for permissions to propagate"
    echo "2. Re-run your pipeline"
    echo "3. If it still fails, try restarting the compute cluster"
    echo ""
else
    echo "ERROR: Failed to assign role"
    echo ""
    echo "You may need to:"
    echo "1. Have 'User Access Administrator' or 'Owner' role on the workspace"
    echo "2. Check if there are policy restrictions preventing role assignments"
    echo ""
    echo "Manual steps:"
    echo "1. Go to Azure Portal"
    echo "2. Navigate to: $WORKSPACE_NAME → Access Control (IAM)"
    echo "3. Click 'Add role assignment'"
    echo "4. Select role: 'AzureML Data Scientist'"
    echo "5. Assign to: Managed Identity → $COMPUTE_NAME"
    exit 1
fi
