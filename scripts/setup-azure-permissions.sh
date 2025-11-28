#!/bin/bash
# =============================================================================
# Azure MLOps Setup Script
# =============================================================================
# This script ensures your Azure service principal has the correct permissions
# to run the full MLOps pipeline, including assigning roles to compute identities.
#
# Run this script once before your first pipeline run, or after creating a new
# Azure environment.
#
# Usage: ./scripts/setup-azure-permissions.sh
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration - should match your workflow
RESOURCE_GROUP="${RESOURCE_GROUP:-mlops-examen-rg}"
LOCATION="${LOCATION:-westeurope}"
SUBSCRIPTION_ID="${SUBSCRIPTION_ID:-}"

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║        🔧 Azure MLOps Permissions Setup                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo -e "${RED}❌ Azure CLI is not installed. Please install it first:${NC}"
    echo "   https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Check if logged in
echo "🔍 Checking Azure CLI login status..."
if ! az account show &> /dev/null; then
    echo -e "${YELLOW}⚠️  Not logged into Azure CLI. Please log in...${NC}"
    az login
fi

CURRENT_USER=$(az account show --query user.name -o tsv)
echo -e "${GREEN}✅ Logged in as: $CURRENT_USER${NC}"

# Get subscription
if [ -z "$SUBSCRIPTION_ID" ]; then
    SUBSCRIPTION_ID=$(az account show --query id -o tsv)
fi
SUBSCRIPTION_NAME=$(az account show --query name -o tsv)
echo "📌 Subscription: $SUBSCRIPTION_NAME ($SUBSCRIPTION_ID)"

# Check if AZURE_CREDENTIALS file or environment variable exists
echo ""
echo "🔍 Looking for service principal credentials..."

CREDENTIALS_FILE=".azure-credentials.json"
SP_CLIENT_ID=""

if [ -f "$CREDENTIALS_FILE" ]; then
    echo "   Found credentials file: $CREDENTIALS_FILE"
    SP_CLIENT_ID=$(cat "$CREDENTIALS_FILE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('clientId', ''))" 2>/dev/null || echo "")
elif [ -n "$AZURE_CREDENTIALS" ]; then
    echo "   Found AZURE_CREDENTIALS environment variable"
    SP_CLIENT_ID=$(echo "$AZURE_CREDENTIALS" | python3 -c "import sys, json; print(json.load(sys.stdin).get('clientId', ''))" 2>/dev/null || echo "")
fi

if [ -z "$SP_CLIENT_ID" ]; then
    echo -e "${YELLOW}"
    echo "⚠️  No service principal credentials found."
    echo ""
    echo "   Please enter the clientId from your GitHub secret AZURE_CREDENTIALS,"
    echo "   or provide the path to a JSON file containing the credentials."
    echo -e "${NC}"
    read -p "Enter Service Principal Client ID (or 'q' to quit): " SP_CLIENT_ID

    if [ "$SP_CLIENT_ID" = "q" ] || [ -z "$SP_CLIENT_ID" ]; then
        echo "Exiting..."
        exit 0
    fi
fi

echo "📌 Service Principal Client ID: $SP_CLIENT_ID"

# Get the Object ID of the service principal
echo ""
echo "🔍 Getting service principal details..."
SP_OBJECT_ID=$(az ad sp show --id "$SP_CLIENT_ID" --query id -o tsv 2>/dev/null || echo "")

if [ -z "$SP_OBJECT_ID" ]; then
    echo -e "${RED}❌ Could not find service principal with Client ID: $SP_CLIENT_ID${NC}"
    echo "   Make sure the Client ID is correct and you have permissions to view it."
    exit 1
fi

SP_DISPLAY_NAME=$(az ad sp show --id "$SP_CLIENT_ID" --query displayName -o tsv)
echo -e "${GREEN}✅ Found: $SP_DISPLAY_NAME${NC}"
echo "   Object ID: $SP_OBJECT_ID"

# Check current role assignments
echo ""
echo "🔍 Checking current role assignments..."
CURRENT_ROLES=$(az role assignment list \
    --assignee "$SP_OBJECT_ID" \
    --scope "/subscriptions/$SUBSCRIPTION_ID" \
    --query "[].roleDefinitionName" -o tsv 2>/dev/null || echo "")

echo "   Current roles at subscription level:"
if [ -z "$CURRENT_ROLES" ]; then
    echo "   (none)"
else
    echo "$CURRENT_ROLES" | while read role; do
        echo "   - $role"
    done
fi

# Check if has Owner or User Access Administrator
HAS_OWNER=$(echo "$CURRENT_ROLES" | grep -c "Owner" || echo "0")
HAS_UAA=$(echo "$CURRENT_ROLES" | grep -c "User Access Administrator" || echo "0")

echo ""
if [ "$HAS_OWNER" -gt 0 ]; then
    echo -e "${GREEN}✅ Service principal already has 'Owner' role - no changes needed!${NC}"
    NEEDS_ROLE=false
elif [ "$HAS_UAA" -gt 0 ]; then
    echo -e "${GREEN}✅ Service principal already has 'User Access Administrator' role - no changes needed!${NC}"
    NEEDS_ROLE=false
else
    echo -e "${YELLOW}⚠️  Service principal lacks permission to assign roles${NC}"
    NEEDS_ROLE=true
fi

# Create resource group if it doesn't exist
echo ""
echo "🔍 Checking resource group..."
if az group show --name "$RESOURCE_GROUP" &>/dev/null; then
    echo -e "${GREEN}✅ Resource group '$RESOURCE_GROUP' exists${NC}"
else
    echo "🔨 Creating resource group '$RESOURCE_GROUP' in '$LOCATION'..."
    az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none
    echo -e "${GREEN}✅ Resource group created${NC}"
fi

RG_SCOPE="/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP"

# Assign roles if needed
if [ "$NEEDS_ROLE" = true ]; then
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "The service principal needs additional permissions to:"
    echo "  - Assign 'AzureML Data Scientist' role to compute cluster identity"
    echo ""
    echo "Options:"
    echo "  1. Assign 'Owner' role on resource group (recommended)"
    echo "  2. Assign 'User Access Administrator' role on resource group"
    echo "  3. Skip (pipeline may fail on role assignment)"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    read -p "Choose option (1/2/3): " CHOICE

    case $CHOICE in
        1)
            echo ""
            echo "🔨 Assigning 'Owner' role on resource group..."
            if az role assignment create \
                --assignee-object-id "$SP_OBJECT_ID" \
                --assignee-principal-type ServicePrincipal \
                --role "Owner" \
                --scope "$RG_SCOPE" \
                --output none 2>&1; then
                echo -e "${GREEN}✅ 'Owner' role assigned successfully!${NC}"
            else
                echo -e "${RED}❌ Failed to assign role. You may not have sufficient permissions.${NC}"
                echo "   Ask your Azure administrator to run this command:"
                echo ""
                echo "   az role assignment create \\"
                echo "     --assignee-object-id $SP_OBJECT_ID \\"
                echo "     --assignee-principal-type ServicePrincipal \\"
                echo "     --role Owner \\"
                echo "     --scope $RG_SCOPE"
                exit 1
            fi
            ;;
        2)
            echo ""
            echo "🔨 Assigning 'User Access Administrator' role on resource group..."
            if az role assignment create \
                --assignee-object-id "$SP_OBJECT_ID" \
                --assignee-principal-type ServicePrincipal \
                --role "User Access Administrator" \
                --scope "$RG_SCOPE" \
                --output none 2>&1; then
                echo -e "${GREEN}✅ 'User Access Administrator' role assigned successfully!${NC}"
            else
                echo -e "${RED}❌ Failed to assign role. You may not have sufficient permissions.${NC}"
                exit 1
            fi
            ;;
        3)
            echo -e "${YELLOW}⚠️  Skipping role assignment. The pipeline may fail.${NC}"
            ;;
        *)
            echo "Invalid option. Exiting."
            exit 1
            ;;
    esac
fi

# Also ensure Contributor role exists on resource group
echo ""
echo "🔍 Checking 'Contributor' role on resource group..."
RG_ROLES=$(az role assignment list \
    --assignee "$SP_OBJECT_ID" \
    --scope "$RG_SCOPE" \
    --query "[].roleDefinitionName" -o tsv 2>/dev/null || echo "")

HAS_CONTRIBUTOR=$(echo "$RG_ROLES" | grep -c "Contributor" || echo "0")
HAS_OWNER_RG=$(echo "$RG_ROLES" | grep -c "Owner" || echo "0")

if [ "$HAS_OWNER_RG" -gt 0 ]; then
    echo -e "${GREEN}✅ Has 'Owner' role (includes Contributor permissions)${NC}"
elif [ "$HAS_CONTRIBUTOR" -gt 0 ]; then
    echo -e "${GREEN}✅ Has 'Contributor' role${NC}"
else
    echo "🔨 Assigning 'Contributor' role on resource group..."
    az role assignment create \
        --assignee-object-id "$SP_OBJECT_ID" \
        --assignee-principal-type ServicePrincipal \
        --role "Contributor" \
        --scope "$RG_SCOPE" \
        --output none 2>&1 || echo -e "${YELLOW}⚠️  Could not assign Contributor role${NC}"
    echo -e "${GREEN}✅ 'Contributor' role assigned${NC}"
fi

# Summary
echo ""
echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║        ✅ Setup Complete!                                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""
echo "Your service principal is now configured with the necessary"
echo "permissions to run the MLOps pipeline."
echo ""
echo "Final role assignments for $SP_DISPLAY_NAME:"
az role assignment list \
    --assignee "$SP_OBJECT_ID" \
    --scope "$RG_SCOPE" \
    --query "[].{Role:roleDefinitionName, Scope:scope}" \
    --output table 2>/dev/null || echo "(Could not list roles)"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Commit and push your changes"
echo "  2. The GitHub Actions workflow will handle everything else!"
echo ""
