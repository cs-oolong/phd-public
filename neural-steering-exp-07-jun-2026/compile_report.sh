#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
typst compile --font-path "$root/fonts" "$root/REPORT.typ" "$root/REPORT.pdf"
