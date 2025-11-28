#!/bin/bash
# =============================================================================
# Install Git Hooks for MLOps Project
# =============================================================================
# This script installs the pre-push hook that verifies Azure permissions
# before pushing code changes.
#
# Usage: ./scripts/install-hooks.sh
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║        🔧 Installing Git Hooks                             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Find the git root directory
GIT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)

if [ -z "$GIT_ROOT" ]; then
    echo -e "${RED}❌ Error: Not inside a git repository${NC}"
    exit 1
fi

# Find the scripts directory (relative to this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create hooks directory if it doesn't exist
HOOKS_DIR="$GIT_ROOT/.git/hooks"
mkdir -p "$HOOKS_DIR"

# Install pre-push hook
echo "📦 Installing pre-push hook..."

PRE_PUSH_SOURCE="$SCRIPT_DIR/pre-push"
PRE_PUSH_DEST="$HOOKS_DIR/pre-push"

if [ -f "$PRE_PUSH_SOURCE" ]; then
    cp "$PRE_PUSH_SOURCE" "$PRE_PUSH_DEST"
    chmod +x "$PRE_PUSH_DEST"
    echo -e "${GREEN}✅ pre-push hook installed${NC}"
else
    echo -e "${RED}❌ Error: pre-push hook source not found at $PRE_PUSH_SOURCE${NC}"
    exit 1
fi

# Make setup script executable
SETUP_SCRIPT="$SCRIPT_DIR/setup-azure-permissions.sh"
if [ -f "$SETUP_SCRIPT" ]; then
    chmod +x "$SETUP_SCRIPT"
    echo -e "${GREEN}✅ setup-azure-permissions.sh made executable${NC}"
fi

echo ""
echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║        ✅ Git Hooks Installed Successfully!                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""
echo "Installed hooks:"
echo "  • pre-push: Checks Azure permissions before pushing"
echo ""
echo -e "${BLUE}What happens now:${NC}"
echo "  • Before each 'git push', the hook will verify your Azure"
echo "    service principal has the correct permissions"
echo "  • If permissions are missing, you'll be prompted to fix them"
echo "  • Checks are cached for 24 hours to avoid slowdowns"
echo ""
echo -e "${BLUE}Optional setup:${NC}"
echo "  • Create .azure-credentials.json with your service principal"
echo "  • Or set AZURE_CLIENT_ID environment variable"
echo "  • Run ./scripts/setup-azure-permissions.sh for initial setup"
echo ""
