#!/bin/bash
# Cluster GPU job planner — shows node availability, running/pending jobs,
# and fit analysis for a typical job request.
#
# Usage:
#   ./job_planner.sh
#   ./job_planner.sh --gpus 2 --mem 64G --cpus 8

set -uo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
REQ_GPUS=1
REQ_MEM="32G"
REQ_CPUS=4

# ── Colors (tput with ANSI fallback) ─────────────────────────────────────────
if [[ -t 1 ]] && command -v tput &>/dev/null && tput setaf 1 &>/dev/null; then
    RED=$(tput setaf 1)
    GREEN=$(tput setaf 2)
    YELLOW=$(tput setaf 3)
    CYAN=$(tput setaf 6)
    BOLD=$(tput bold)
    DIM=$(tput dim)
    RESET=$(tput sgr0)
else
    RED=$'\033[0;31m'
    GREEN=$'\033[0;32m'
    YELLOW=$'\033[0;33m'
    CYAN=$'\033[0;36m'
    BOLD=$'\033[1m'
    DIM=$'\033[2m'
    RESET=$'\033[0m'
fi

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)  REQ_GPUS="$2"; shift 2 ;;
        --mem)   REQ_MEM="$2";  shift 2 ;;
        --cpus)  REQ_CPUS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--gpus N] [--mem XG] [--cpus N]"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── GPU VRAM lookup (SLURM does not report this) ────────────────────────────
gpu_vram_gb() {
    case "$1" in
        L40S|l40s)     echo 48 ;;
        H100|h100)     echo 80 ;;
        RTX8000)       echo 48 ;;
        RTXA5000|A5000) echo 24 ;;
        RTX5000)       echo 16 ;;
        P5000)         echo 16 ;;
        *)             echo 0 ;;
    esac
}

# Human-readable GPU label for display
gpu_label() {
    case "$1" in
        L40S)      echo "L40S" ;;
        H100)      echo "H100" ;;
        RTX8000)   echo "RTX 8000" ;;
        RTXA5000)  echo "RTX A5000" ;;
        RTX5000)   echo "RTX 5000" ;;
        P5000)     echo "P5000" ;;
        *)         echo "$1" ;;
    esac
}

# ── Memory helpers ────────────────────────────────────────────────────────────
mem_to_mb() {
    local val="${1// /}"
    if [[ "$val" =~ ^([0-9]+)[Gg]$ ]]; then
        echo $(( BASH_REMATCH[1] * 1024 ))
    elif [[ "$val" =~ ^([0-9]+)[Mm]$ ]]; then
        echo "${BASH_REMATCH[1]}"
    elif [[ "$val" =~ ^([0-9]+)$ ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo 0
    fi
}

format_mem_gb() {
    local mb="$1"
    if [[ "$mb" -ge 1024 ]]; then
        echo "$(( mb / 1024 ))G"
    else
        echo "${mb}M"
    fi
}

# ── SLURM time helpers ────────────────────────────────────────────────────────
time_to_seconds() {
    local t="$1"
    local days=0 hours=0 mins=0 secs=0
    if [[ "$t" =~ ^([0-9]+)-([0-9]+):([0-9]+):([0-9]+)$ ]]; then
        days=$((10#${BASH_REMATCH[1]})); hours=$((10#${BASH_REMATCH[2]}))
        mins=$((10#${BASH_REMATCH[3]})); secs=$((10#${BASH_REMATCH[4]}))
    elif [[ "$t" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
        hours=$((10#${BASH_REMATCH[1]})); mins=$((10#${BASH_REMATCH[2]})); secs=$((10#${BASH_REMATCH[3]}))
    elif [[ "$t" =~ ^([0-9]+):([0-9]+)$ ]]; then
        mins=$((10#${BASH_REMATCH[1]})); secs=$((10#${BASH_REMATCH[2]}))
    else
        echo 0; return
    fi
    echo $(( days*86400 + hours*3600 + mins*60 + secs ))
}

format_remaining() {
    local limit="$1" elapsed="$2"
    local lim_sec elapsed_sec rem
    lim_sec=$(time_to_seconds "$limit")
    elapsed_sec=$(time_to_seconds "$elapsed")
    rem=$(( lim_sec - elapsed_sec ))
    if [[ "$rem" -le 0 ]]; then
        echo "expired"
        return
    fi
    local d=$(( rem / 86400 )); rem=$(( rem % 86400 ))
    local h=$(( rem / 3600 )); rem=$(( rem % 3600 ))
    local m=$(( rem / 60 ))
    if [[ "$d" -gt 0 ]]; then
        printf "%d-%02d:%02d left" "$d" "$h" "$m"
    else
        printf "%02d:%02d:%02d left" "$h" "$m" "$(( rem % 60 ))"
    fi
}

# ── Parse scontrol field ──────────────────────────────────────────────────────
sc_field() {
    local data="$1" key="$2"
    echo "$data" | grep -oP "${key}=\K[^ ]+" | head -1
}

# Parse Gres=gpu:TYPE:COUNT → returns "TYPE COUNT"
parse_gres_total() {
    local gres="$1"
    if [[ "$gres" =~ gpu:([^:]+):([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]}"
    elif [[ "$gres" =~ gpu:([0-9]+) ]]; then
        echo "gpu ${BASH_REMATCH[1]}"
    else
        echo "unknown 0"
    fi
}

# Parse GresUsed=gpu:TYPE:N(IDX:...) or gpu:N
parse_gres_used_field() {
    local used="$1"
    local total=0
    if [[ -z "$used" || "$used" == "(null)" ]]; then
        echo 0; return
    fi
    # Handle comma-separated multiple GRES entries
    local entry
    IFS=',' read -ra entries <<< "$used"
    for entry in "${entries[@]}"; do
        if [[ "$entry" =~ gpu:[^:]*:([0-9]+) ]]; then
            total=$(( total + BASH_REMATCH[1] ))
        elif [[ "$entry" =~ gpu:([0-9]+) ]]; then
            total=$(( total + BASH_REMATCH[1] ))
        fi
    done
    echo "$total"
}

# Parse gpu count from AllocTRES=...,gres/gpu=3,...
parse_alloc_tres_gpus() {
    local alloc="$1"
    if [[ "$alloc" =~ gres/gpu[^=]*=([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo 0
    fi
}

# Sum GPUs from running jobs on a node via squeue
sum_gpus_from_jobs() {
    local node="$1"
    local total=0 val
    while IFS= read -r tres; do
        [[ -z "$tres" || "$tres" == "N/A" ]] && continue
        if [[ "$tres" =~ gres/gpu:[^:]*:([0-9]+) ]]; then
            total=$(( total + BASH_REMATCH[1] ))
        elif [[ "$tres" =~ gres/gpu:([0-9]+) ]]; then
            total=$(( total + BASH_REMATCH[1] ))
        fi
    done < <(squeue -w "$node" -h -o "%b" --states=R 2>/dev/null)
    echo "$total"
}

# ── Fit / recommendation text ─────────────────────────────────────────────────
vram_advice() {
    local vram="$1"
    if [[ "$vram" -ge 48 ]]; then
        echo "model fits in full precision"
    elif [[ "$vram" -ge 24 ]]; then
        echo "fp16 OK, int8 for larger models"
    elif [[ "$vram" -ge 16 ]]; then
        echo "needs int8 quantization"
    else
        echo "limited VRAM"
    fi
}

node_status() {
    local free_gpu="$1" free_mem_mb="$2" free_cpu="$3"
    local req_mem_mb
    req_mem_mb=$(mem_to_mb "$REQ_MEM")

    if [[ "$free_gpu" -lt "$REQ_GPUS" ]]; then
        echo "full:no_gpu"
    elif [[ "$free_mem_mb" -lt "$req_mem_mb" ]]; then
        echo "full:no_mem"
    elif [[ "$free_cpu" -lt "$REQ_CPUS" ]]; then
        echo "full:no_cpu"
    elif [[ "$free_mem_mb" -lt $(( req_mem_mb * 2 )) || "$free_cpu" -lt $(( REQ_CPUS * 2 )) ]]; then
        echo "tight"
    else
        echo "ok"
    fi
}

status_display() {
    local status="$1"
    case "$status" in
        ok)       echo -e "${GREEN}✓ Available${RESET}" ;;
        tight)    echo -e "${YELLOW}⚠ Tight${RESET}" ;;
        full:*)   echo -e "${RED}✗ Full${RESET}" ;;
        *)        echo -e "${RED}✗ Full${RESET}" ;;
    esac
}

status_reason_short() {
    local status="$1" free_gpu="$2" tot_gpu="$3"
    case "$status" in
        full:no_gpu) echo "No GPU free" ;;
        full:no_mem) echo "Insufficient RAM" ;;
        full:no_cpu) echo "Insufficient CPUs" ;;
        tight)       echo "Low headroom" ;;
        ok)          echo "${free_gpu}/${tot_gpu} GPUs free" ;;
    esac
}

# ── Prefetch running jobs GPU usage per node ──────────────────────────────────
declare -A NODE_GPU_FROM_JOBS=()
while IFS=' ' read -r node tres; do
    [[ -z "$node" || -z "$tres" || "$tres" == "N/A" ]] && continue
    local_count=0
    if [[ "$tres" =~ gres/gpu:[^:]*:([0-9]+) ]]; then
        local_count=${BASH_REMATCH[1]}
    elif [[ "$tres" =~ gres/gpu:([0-9]+) ]]; then
        local_count=${BASH_REMATCH[1]}
    fi
    NODE_GPU_FROM_JOBS["$node"]=$(( ${NODE_GPU_FROM_JOBS[$node]:-0} + local_count ))
done < <(squeue -h -o "%N %b" --states=R 2>/dev/null | tr ',' '\n' | awk '{print $1, $2}')

# ── Collect node data ─────────────────────────────────────────────────────────
declare -a NODES_DATA=()

while IFS=' ' read -r node partition gres state; do
    [[ -z "$node" ]] && continue
    # Strip trailing * from partition name (default marker)
    partition="${partition%\*}"

    sc_out=$(scontrol show node "$node" 2>/dev/null) || continue
    [[ -z "$sc_out" ]] && continue

    cpu_tot=$(sc_field "$sc_out" "CPUTot")
    cpu_alloc=$(sc_field "$sc_out" "CPUAlloc")
    real_mem=$(sc_field "$sc_out" "RealMemory")
    alloc_mem=$(sc_field "$sc_out" "AllocMem")
    gres_field=$(sc_field "$sc_out" "Gres")
    gres_used_field=$(sc_field "$sc_out" "GresUsed")
    alloc_tres=$(sc_field "$sc_out" "AllocTRES")

    read -r gpu_type tot_gpu <<< "$(parse_gres_total "$gres_field")"

    # Determine allocated GPUs: GresUsed → AllocTRES → squeue sum
    gpu_alloc=0
    if [[ -n "$gres_used_field" && "$gres_used_field" != "(null)" ]]; then
        gpu_alloc=$(parse_gres_used_field "$gres_used_field")
    fi
    if [[ "$gpu_alloc" -eq 0 ]]; then
        gpu_alloc=$(parse_alloc_tres_gpus "$alloc_tres")
    fi
    if [[ "$gpu_alloc" -eq 0 && -n "${NODE_GPU_FROM_JOBS[$node]:-}" ]]; then
        gpu_alloc=${NODE_GPU_FROM_JOBS[$node]}
    fi

    cpu_free=$(( cpu_tot - cpu_alloc ))
    mem_free=$(( real_mem - alloc_mem ))
    gpu_free=$(( tot_gpu - gpu_alloc ))
    [[ "$gpu_free" -lt 0 ]] && gpu_free=0

    vram=$(gpu_vram_gb "$gpu_type")
    st=$(node_status "$gpu_free" "$mem_free" "$cpu_free")

    NODES_DATA+=("$partition|$node|$gpu_type|$vram|$gpu_free|$tot_gpu|$mem_free|$real_mem|$cpu_free|$cpu_tot|$st")
done < <(sinfo -N -h -o "%N %P %G %T" 2>/dev/null | grep -i gpu)

# ── Header ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║${RESET}              ${CYAN}Cluster GPU Planner${RESET}                              ${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}              $(date '+%Y-%m-%d %H:%M:%S')                              ${BOLD}║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "Your job request: ${BOLD}${REQ_GPUS} GPU${RESET} | ${BOLD}${REQ_MEM} RAM${RESET} | ${BOLD}${REQ_CPUS} CPUs${RESET}"
echo ""

# ── Section 1: Node Availability ──────────────────────────────────────────────
echo -e "${BOLD}── Node Availability ${DIM}────────────────────────────────────────────${RESET}"
printf "${BOLD}%-10s %-6s %-12s %5s  %-18s %-18s %-18s %s${RESET}\n" \
    "PARTITION" "NODE" "GPU TYPE" "VRAM" "GPUs (free/total)" "RAM (free/total)" "CPUs (free/total)" "STATUS"
echo "${DIM}$(printf '%.0s─' {1..110})${RESET}"

for entry in "${NODES_DATA[@]}"; do
    IFS='|' read -r part node gtype vram gfree gtot mfree mtot cfree ctot st <<< "$entry"
    glab=$(gpu_label "$gtype")
    gstr="${gfree}/${gtot}"
    mstr="$(format_mem_gb "$mfree")/$(format_mem_gb "$mtot")"
    cstr="${cfree}/${ctot}"
    sd=$(status_display "$st")

    printf "%-10s %-6s %-12s %4sGB  %-18s %-18s %-18s " \
        "$part" "$node" "$glab" "$vram" "$gstr" "$mstr" "$cstr"
    echo -e "$sd"
done
echo ""

# ── Section 2: Running Jobs ───────────────────────────────────────────────────
echo -e "${BOLD}── Running Jobs ${DIM}─────────────────────────────────────────────────${RESET}"
running=$(squeue -o "%.18i %.12P %.8u %.2t %.12M %.12l %.4C %.7m %b %N" --states=R 2>/dev/null)
if [[ -z "$running" ]] || [[ $(echo "$running" | wc -l) -le 1 ]]; then
    echo "  (no running jobs)"
else
    printf "${BOLD}%-18s %-12s %-8s %-12s %-12s %4s %7s %-16s %s${RESET}\n" \
        "JOBID" "PARTITION" "USER" "RUNTIME" "REMAINING" "CPU" "MEM" "GPUS" "NODE"
    echo "${DIM}$(printf '%.0s─' {1..100})${RESET}"

    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*JOBID ]] && continue
        [[ -z "${line// /}" ]] && continue

        jobid=$(echo "$line" | awk '{print $1}')
        part=$(echo "$line" | awk '{print $2}')
        user=$(echo "$line" | awk '{print $3}')
        runtime=$(echo "$line" | awk '{print $5}')
        timelim=$(echo "$line" | awk '{print $6}')
        cpus=$(echo "$line" | awk '{print $7}')
        mem=$(echo "$line" | awk '{print $8}')
        gres=$(echo "$line" | awk '{print $9}')
        nodelist=$(echo "$line" | awk '{print $10}')

        remaining=$(format_remaining "$timelim" "$runtime")
        [[ "$gres" == "N/A" ]] && gres="—"

        printf "%-18s %-12s %-8s %-12s %-12s %4s %7s %-16s %s\n" \
            "$jobid" "$part" "$user" "$runtime" "$remaining" "$cpus" "$mem" "$gres" "$nodelist"
    done <<< "$running"
fi
echo ""

# ── Section 3: Recommendations ────────────────────────────────────────────────
echo -e "${BOLD}── Recommendation ${DIM}───────────────────────────────────────────────${RESET}"
echo -e "Best options for your job (${REQ_GPUS} GPU, ${REQ_MEM} RAM, ${REQ_CPUS} CPUs):"
echo ""

# Build sorted recommendations: available nodes sorted by vram desc, then free gpus desc
declare -a RECS=()
declare -a NONFIT=()

for entry in "${NODES_DATA[@]}"; do
    IFS='|' read -r part node gtype vram gfree gtot mfree mtot cfree ctot st <<< "$entry"
    glab=$(gpu_label "$gtype")
    advice=$(vram_advice "$vram")

    if [[ "$st" == "ok" || "$st" == "tight" ]]; then
        # Score: vram*1000 + gfree*10 + (tight=0, ok=1)
        tight_bonus=0
        [[ "$st" == "ok" ]] && tight_bonus=1
        score=$(( vram * 1000 + gfree * 10 + tight_bonus ))
        RECS+=("${score}|${part}|${node}|${glab}|${vram}|${advice}|${gfree}|${st}")
    else
        reason=$(status_reason_short "$st" "$gfree" "$gtot")
        NONFIT+=("${part}|${node}|${glab}|${reason}")
    fi
done

if [[ ${#RECS[@]} -eq 0 ]]; then
    echo -e "  ${RED}No nodes can fit your job right now.${RESET}"
    echo ""
    echo "  Nodes that cannot fit:"
    for nf in "${NONFIT[@]}"; do
        IFS='|' read -r part node glab reason <<< "$nf"
        echo -e "    ${RED}✗${RESET} ${part} / ${node} — ${glab}: ${reason}"
    done
else
    IFS=$'\n' sorted=($(printf '%s\n' "${RECS[@]}" | sort -t'|' -k1 -nr))
    rank=1
    for rec in "${sorted[@]}"; do
        IFS='|' read -r _ part node glab vram advice gfree st <<< "$rec"
        color="$GREEN"
        [[ "$st" == "tight" ]] && color="$YELLOW"
        echo -e "  ${color}${rank}.${RESET} ${part} / ${node}  — ${glab} ${vram}GB: ${advice}, ${gfree} GPU(s) free"
        rank=$(( rank + 1 ))
    done

    if [[ ${#NONFIT[@]} -gt 0 ]]; then
        echo ""
        echo "  Cannot fit now:"
        for nf in "${NONFIT[@]}"; do
            IFS='|' read -r part node glab reason <<< "$nf"
            echo -e "    ${RED}✗${RESET} ${part} / ${node} — ${glab}: ${reason}"
        done
    fi
fi
echo ""

# ── Section 4: Pending Jobs ─────────────────────────────────────────────────
echo -e "${BOLD}── Pending Jobs ${DIM}─────────────────────────────────────────────────${RESET}"
pending=$(squeue -o "%.18i %.12P %.8u %.4C %.7m %b %r" --states=PD 2>/dev/null)
if [[ -z "$pending" ]] || [[ $(echo "$pending" | wc -l) -le 1 ]]; then
    echo "  (no pending jobs)"
else
    printf "${BOLD}%-18s %-12s %-8s %4s %7s %-14s %s${RESET}\n" \
        "JOBID" "PARTITION" "USER" "CPU" "MEM" "GPUS" "REASON"
    echo "${DIM}$(printf '%.0s─' {1..90})${RESET}"

    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*JOBID ]] && continue
        [[ -z "${line// /}" ]] && continue

        jobid=$(echo "$line" | awk '{print $1}')
        part=$(echo "$line" | awk '{print $2}')
        user=$(echo "$line" | awk '{print $3}')
        cpus=$(echo "$line" | awk '{print $4}')
        mem=$(echo "$line" | awk '{print $5}')
        gres=$(echo "$line" | awk '{print $6}')
        reason=$(echo "$line" | awk '{print $7}')
        [[ "$gres" == "N/A" ]] && gres="—"

        printf "%-18s %-12s %-8s %4s %7s %-14s %s\n" \
            "$jobid" "$part" "$user" "$cpus" "$mem" "$gres" "$reason"
    done <<< "$pending"
fi
echo ""
