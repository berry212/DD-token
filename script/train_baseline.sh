#!/usr/bin/env sh
set -eu

FIRST_ARG="${1:-vit}"
MODEL="vit"
DATASET="pathmnist"

case "$FIRST_ARG" in
    pathmnist|dermamnist|derma|dermamnist+)
        MODEL="vit"
        DATASET="$FIRST_ARG"
        ;;
    *)
        MODEL="$FIRST_ARG"
        DATASET="${2:-pathmnist}"
        ;;
esac

case "$MODEL" in
    vit)
        sh script/train_baseline_vit.sh "$DATASET"
        ;;
    resnet)
        sh script/train_baseline_resnet.sh "$DATASET"
        ;;
    mlp)
        sh script/train_baseline_mlp.sh "$DATASET"
        ;;
    *)
        echo "Usage: sh script/train_baseline.sh [vit|resnet|mlp] [pathmnist|dermamnist]" >&2
        exit 1
        ;;
esac
