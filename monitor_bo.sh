#!/bin/bash
while ps aux | grep -q "[p]4tun.bo.optimize"; do
    echo "=== $(date) ==="
    ps aux | grep "[p]4tun.bo.optimize" | head -1
    ls -lh p4tun/bo/results/*2-2*sam*checkpoint* 2>/dev/null | tail -1
    ls -lh p4tun/bo/results/*2-2*sam*.json 2>/dev/null | tail -3
    echo ""
    sleep 300
done
echo "BO completed!"
ls -lh p4tun/bo/results/*2-2*sam*.json 2>/dev/null | tail -3
