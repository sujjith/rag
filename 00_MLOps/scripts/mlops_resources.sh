#!/bin/bash
# MLOps Stack Resource Monitor
# Usage: ./mlops_resources.sh

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                  MLOps Platform Resource Usage                    ║"
echo "║                  $(date '+%Y-%m-%d %H:%M:%S')                          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Node Resources
echo "┌──────────────────────────────────────────────────────────────────┐"
echo "│ NODE RESOURCES                                                    │"
echo "└──────────────────────────────────────────────────────────────────┘"
kubectl top nodes
echo ""

# K3s Process
echo "┌──────────────────────────────────────────────────────────────────┐"
echo "│ K3s SERVER PROCESS                                               │"
echo "└──────────────────────────────────────────────────────────────────┘"
ps aux | grep "/usr/local/bin/k3s" | grep -v grep | awk '{printf "  CPU: %s%%  |  Memory: %.1f GB (RSS: %s KB)\n", $3, $6/1024/1024, $6}'
echo ""

# Total Pod Resources
echo "┌──────────────────────────────────────────────────────────────────┐"
echo "│ TOTAL POD RESOURCES                                              │"
echo "└──────────────────────────────────────────────────────────────────┘"
kubectl top pods -A --no-headers 2>/dev/null | awk '
{
    cpu=$3; mem=$4
    gsub(/m/, "", cpu)
    gsub(/Mi/, "", mem)
    total_cpu+=cpu
    total_mem+=mem
    count++
}
END {
    printf "  Total Pods: %d\n", count
    printf "  Total CPU:  %dm (%.2f vCPU)\n", total_cpu, total_cpu/1000
    printf "  Total RAM:  %dMi (%.1f GB)\n", total_mem, total_mem/1024
}'
echo ""

# Resource by Namespace
echo "┌──────────────────────────────────────────────────────────────────┐"
echo "│ RESOURCES BY NAMESPACE (sorted by memory)                        │"
echo "└──────────────────────────────────────────────────────────────────┘"
printf "  %-20s %8s %10s %6s\n" "NAMESPACE" "CPU" "MEMORY" "PODS"
echo "  ────────────────────────────────────────────────────────"
kubectl top pods -A --no-headers 2>/dev/null | awk '
{
    ns=$1; cpu=$3; mem=$4
    gsub(/m/, "", cpu)
    gsub(/Mi/, "", mem)
    ns_cpu[ns]+=cpu
    ns_mem[ns]+=mem
    ns_count[ns]++
}
END {
    for (n in ns_cpu) {
        printf "  %-20s %6dm %8dMi %6d\n", n, ns_cpu[n], ns_mem[n], ns_count[n]
    }
}' | sort -t'i' -k3 -rn
echo ""

# Top 10 Pods by Memory
echo "┌──────────────────────────────────────────────────────────────────┐"
echo "│ TOP 10 PODS BY MEMORY                                            │"
echo "└──────────────────────────────────────────────────────────────────┘"
printf "  %-20s %-40s %8s\n" "NAMESPACE" "POD" "MEMORY"
echo "  ────────────────────────────────────────────────────────────────"
kubectl top pods -A --no-headers --sort-by=memory 2>/dev/null | head -10 | awk '{printf "  %-20s %-40s %8s\n", $1, $2, $4}'
echo ""

# Service Endpoints
echo "┌──────────────────────────────────────────────────────────────────┐"
echo "│ SERVICE ENDPOINTS                                                │"
echo "└──────────────────────────────────────────────────────────────────┘"
kubectl get svc -A --field-selector spec.type=NodePort -o custom-columns='NAMESPACE:.metadata.namespace,NAME:.metadata.name,PORT:.spec.ports[0].nodePort' --no-headers 2>/dev/null | awk '{printf "  %-15s %-25s http://localhost:%s\n", $1, $2, $3}'
echo ""

echo "═══════════════════════════════════════════════════════════════════"
echo "  Run 'watch -n 5 ./mlops_resources.sh' for live monitoring"
echo "═══════════════════════════════════════════════════════════════════"
