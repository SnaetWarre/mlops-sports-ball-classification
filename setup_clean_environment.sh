#!/bin/bash

# Variables
RG_NAME="mlops-clean-rg"
WS_NAME="mlops-clean-ws"
LOCATION="westeurope"
COMPUTE_NAME="warre-compute"

# 1. Create Resource Group
echo "Creating Resource Group: $RG_NAME..."
az group create --name $RG_NAME --location $LOCATION

# 2. Create Workspace
echo "Creating Workspace: $WS_NAME..."
az ml workspace create --name $WS_NAME --resource-group $RG_NAME

# 3. Set Defaults
echo "Setting defaults..."
az configure --defaults group=$RG_NAME workspace=$WS_NAME location=$LOCATION

# 4. Create Compute
echo "Creating Compute Instance (this may take a few minutes)..."
# Ensure we are running from the root of the workspace where assignment/ folder is
az ml compute create --file assignment/environment/compute.yaml

# 5. Assign Permissions
echo "Getting Compute Identity..."
PRINCIPAL_ID=$(az ml compute show --name $COMPUTE_NAME --query identity.principal_id -o tsv)
echo "Compute Principal ID: $PRINCIPAL_ID"

echo "Getting Workspace Scope..."
WS_SCOPE=$(az ml workspace show --name $WS_NAME --resource-group $RG_NAME --query id -o tsv)
echo "Workspace Scope: $WS_SCOPE"

echo "Assigning 'Azure Machine Learning Data Scientist' role..."
az role assignment create --assignee $PRINCIPAL_ID --role "f6c7c914-8db3-469d-8ca1-694a8f32e121" --scope $WS_SCOPE

echo "----------------------------------------------------------------"
echo "Done! A clean environment has been set up."
echo "Resource Group: $RG_NAME"
echo "Workspace: $WS_NAME"
echo "Compute: $COMPUTE_NAME"
echo "Permissions have been automatically assigned."
echo "----------------------------------------------------------------"
