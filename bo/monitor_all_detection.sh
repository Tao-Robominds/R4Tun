#!/bin/bash
cd /home/boringtao/Projects/Bayesian-R4Tun

echo "=== Detection BO Monitor ==="
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
  clear
  echo "=== Detection BO Progress $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo ""
  
  # 3-1
  r3_1=$(ls bo/continuous/logs/detect_3-1_*.json 2>/dev/null | wc -l)
  if [ $r3_1 -ge 80 ]; then
    echo "✅ 3-1: COMPLETE ($r3_1/80)"
    # Get best F1
    best=$(venv/bin/python << PYEOF
import json, glob
logs = sorted(glob.glob("bo/continuous/logs/detect_3-1_*.json"))
best_f1 = -1
for lf in logs:
    try:
        with open(lf) as f:
            d = json.load(f)
        f1 = d.get('bo', {}).get('objective_value', 0)
        if f1 > best_f1:
            best_f1 = f1
    except:
        pass
print(f"{best_f1:.4f}")
PYEOF
)
    echo "   Best F1: $best"
  else
    echo "🔄 3-1: $r3_1/80"
  fi
  
  # 4-1
  r4_1=$(ls bo/complex_staggered/logs/detect_4-1_*.json 2>/dev/null | wc -l)
  if [ $r4_1 -ge 80 ]; then
    echo "✅ 4-1: COMPLETE ($r4_1/80)"
    best=$(venv/bin/python << PYEOF
import json, glob
logs = sorted(glob.glob("bo/complex_staggered/logs/detect_4-1_*.json"))
best_f1 = -1
for lf in logs:
    try:
        with open(lf) as f:
            d = json.load(f)
        f1 = d.get('bo', {}).get('objective_value', 0)
        if f1 > best_f1:
            best_f1 = f1
    except:
        pass
print(f"{best_f1:.4f}")
PYEOF
)
    echo "   Best F1: $best"
  else
    echo "🔄 4-1: $r4_1/80"
  fi
  
  # 5-1
  r5_1=$(ls bo/complex_staggered/logs/detect_5-1_*.json 2>/dev/null | wc -l)
  if [ $r5_1 -ge 80 ]; then
    echo "✅ 5-1: COMPLETE ($r5_1/80)"
    best=$(venv/bin/python << PYEOF
import json, glob
logs = sorted(glob.glob("bo/complex_staggered/logs/detect_5-1_*.json"))
best_f1 = -1
for lf in logs:
    try:
        with open(lf) as f:
            d = json.load(f)
        f1 = d.get('bo', {}).get('objective_value', 0)
        if f1 > best_f1:
            best_f1 = f1
    except:
        pass
print(f"{best_f1:.4f}")
PYEOF
)
    echo "   Best F1: $best"
  elif [ $r5_1 -gt 0 ]; then
    echo "🔄 5-1: $r5_1/80"
  else
    echo "⏳ 5-1: Not started yet"
  fi
  
  echo ""
  echo "Active processes: $(ps aux | grep '[r]un_detection_bo.py' | grep -v grep | wc -l)"
  
  # Check if all complete
  if [ $r3_1 -ge 80 ] && [ $r4_1 -ge 80 ] && [ $r5_1 -ge 80 ]; then
    echo ""
    echo "🎉 ALL EXPERIMENTS COMPLETE!"
    break
  fi
  
  sleep 60
done
