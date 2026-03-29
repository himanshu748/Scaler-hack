#!/bin/bash
# Deploy the API Design Environment to HF Spaces
#
# Usage:
#   1. Get a write token from https://huggingface.co/settings/tokens
#   2. Run: HF_TOKEN=hf_xxx ./deploy.sh <your-hf-username>
#
# Example:
#   HF_TOKEN=hf_abc123 ./deploy.sh johndoe
#   -> Deploys to https://huggingface.co/spaces/johndoe/api-design-env

set -e

USERNAME="${1:?Usage: HF_TOKEN=hf_xxx ./deploy.sh <hf-username>}"

if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is required."
    echo "Get one at: https://huggingface.co/settings/tokens"
    exit 1
fi

echo "Logging in to Hugging Face..."
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

echo "Pushing to HF Spaces as ${USERNAME}/api-design-env..."
cd "$(dirname "$0")/api_design_env"
openenv push --repo-id "${USERNAME}/api-design-env"

echo ""
echo "Done! Your environment is live at:"
echo "  Web UI:  https://${USERNAME}-api-design-env.hf.space/web"
echo "  API:     https://${USERNAME}-api-design-env.hf.space/docs"
echo "  Health:  https://${USERNAME}-api-design-env.hf.space/health"
