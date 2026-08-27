#!/usr/bin/env bash
set -euo pipefail

action=${1:-}
if [[ -z "$action" ]]; then
    echo "usage: $0 {discover|build|prepare|audit} [options]" >&2
    exit 2
fi
shift

base=${CER_INTERFACE_BASE:-/ddnB/work/lgutsev/LATech_PROJS/Cer_Interface}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_root=${IFACE_MLFF_SOURCE:-$base/MD_Period/Step2_450K}
campaign_root=${IFACE_MLFF_OUTPUT:-$base/MD_Period/VASP_MLFF_Interfaces}
manifest=${IFACE_MLFF_MANIFEST:-$base/MD_Period/VASP_MLFF_Interfaces_source_manifest.csv}
profile=${IFACE_MLFF_PROFILE:-$script_dir/../examples/mlff-interfaces/profile_loni.yaml}
iface_bin=${IFACE_BIN:-iface}

case "$action" in
    discover)
        mkdir -p "$(dirname "$manifest")"
        exec "$iface_bin" mlff-interfaces discover "$source_root" "$manifest" "$@"
        ;;
    build)
        exec "$iface_bin" mlff-interfaces build "$manifest" "$campaign_root" \
            --profile "$profile" --profile-name vasp_train \
            --encut 520 --ivdw 11 --potim 1.0 \
            --tebeg 300 --teend 600 --train-nsw 3000 --stability-nsw 3000 "$@"
        ;;
    prepare)
        "$iface_bin" prepare -c "$campaign_root/campaign.yaml" "$@"
        exec "$iface_bin" mlff-interfaces array-launch \
            -c "$campaign_root/campaign.yaml" --stage train --concurrency 2 --force
        ;;
    audit)
        exec "$iface_bin" mlff-interfaces audit -c "$campaign_root/campaign.yaml" "$@"
        ;;
    *)
        echo "unknown action: $action" >&2
        echo "usage: $0 {discover|build|prepare|audit} [options]" >&2
        exit 2
        ;;
esac
