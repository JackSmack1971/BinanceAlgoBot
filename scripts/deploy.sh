#!/usr/bin/env bash
set -euo pipefail

ENVIRONMENT=${1:-development}
IMAGE="trading-bot:${ENVIRONMENT}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEPLOY_DIR="$ROOT_DIR/deployment"

# Build image
docker build -t "$IMAGE" -f "$DEPLOY_DIR/docker/Dockerfile" "$ROOT_DIR"
# Security scan
docker run --rm "$IMAGE" pip-audit || { echo "Security scan failed"; exit 1; }

# Apply Kubernetes manifests
kubectl apply -f "$DEPLOY_DIR/kubernetes/configmap.yaml"
kubectl apply -f "$DEPLOY_DIR/kubernetes/secret.yaml"
kubectl apply -f "$DEPLOY_DIR/kubernetes/serviceaccount.yaml"
kubectl apply -f "$DEPLOY_DIR/kubernetes/deployment.yaml"
kubectl apply -f "$DEPLOY_DIR/kubernetes/service.yaml"
kubectl apply -f "$DEPLOY_DIR/kubernetes/hpa.yaml"
kubectl apply -f "$DEPLOY_DIR/kubernetes/networkpolicy.yaml"
kubectl apply -f "$DEPLOY_DIR/kubernetes/servicemonitor.yaml"

# Wait for rollout
if ! kubectl rollout status deployment/trading-bot --timeout=120s; then
  echo "Deployment failed, rolling back"
  kubectl rollout undo deployment/trading-bot
  exit 1
fi

# Smoke test
if ! curl -sf $(kubectl get svc trading-bot -o jsonpath='http://{.spec.clusterIP}:80/healthz'); then
  echo "Smoke test failed, rolling back"
  kubectl rollout undo deployment/trading-bot
  exit 1
fi

echo "Deployment successful"
