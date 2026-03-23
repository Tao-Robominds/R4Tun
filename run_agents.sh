#!/bin/bash

# Script to run all agents step by step for a given tunnel_id
# Usage: ./run_agents.sh <tunnel_id>
# Example: ./run_agents.sh 1-4

# Don't use set -e, we'll handle errors manually

# Check if tunnel_id is provided
if [ $# -eq 0 ]; then
    echo "❌ Error: tunnel_id is required"
    echo "Usage: $0 <tunnel_id>"
    echo "Example: $0 1-4"
    exit 1
fi

TUNNEL_ID=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "=========================================="
echo "🚀 Running Agents Pipeline for Tunnel: $TUNNEL_ID"
echo "=========================================="
echo ""

# Step 0: Create raw characteristics first (required for unfolding analyst)
echo "📊 Step 0: Generating raw characteristics..."
if [ ! -f "data/${TUNNEL_ID}.txt" ]; then
    echo "❌ Error: Input file data/${TUNNEL_ID}.txt not found"
    exit 1
fi

python3 sam4tun/plugins/raw_characteristics.py --tunnel_id "$TUNNEL_ID" || {
    echo "❌ Failed to generate raw characteristics"
    exit 1
}

if [ ! -f "data/ablation/${TUNNEL_ID}/characteristics/raw_characteristics.json" ]; then
    echo "❌ Error: raw_characteristics.json was not created"
    exit 1
fi
echo "✅ Raw characteristics created"
echo ""

# Step 1: Unfolding
echo "=========================================="
echo "🔄 Step 1: Unfolding Agent"
echo "=========================================="
echo "📊 Running unfolding analyst..."
python3 agents/unfolding/analyst.py "$TUNNEL_ID" > "data/${TUNNEL_ID}/analysis/unfolding_analysis.md" || {
    echo "❌ Unfolding analyst failed"
    exit 1
}
echo "✅ Unfolding analysis complete"

echo "💻 Running unfolding coder..."
python3 agents/unfolding/coder.py "$TUNNEL_ID" || {
    echo "❌ Unfolding coder failed"
    exit 1
}
echo "✅ Unfolding complete"
echo ""

# Step 2: Denoising
echo "=========================================="
echo "🔄 Step 2: Denoising Agent"
echo "=========================================="
echo "📊 Running denoising analyst..."
python3 agents/denoising/analyst.py "$TUNNEL_ID" > "data/${TUNNEL_ID}/analysis/denoising_analysis.md" || {
    echo "❌ Denoising analyst failed"
    exit 1
}
echo "✅ Denoising analysis complete"

echo "💻 Running denoising coder..."
python3 agents/denoising/coder.py "$TUNNEL_ID" || {
    echo "❌ Denoising coder failed"
    exit 1
}
echo "✅ Denoising complete"
echo ""

# Step 3: Enhancing
echo "=========================================="
echo "🔄 Step 3: Enhancing Agent"
echo "=========================================="
echo "📊 Running enhancing analyst..."
python3 agents/enhancing/analyst.py "$TUNNEL_ID" > "data/${TUNNEL_ID}/analysis/enhancing_analysis.md" || {
    echo "❌ Enhancing analyst failed"
    exit 1
}
echo "✅ Enhancing analysis complete"

echo "💻 Running enhancing coder..."
if python3 agents/enhancing/coder.py "$TUNNEL_ID"; then
    echo "✅ Enhancing complete"
else
    echo "❌ Enhancing coder failed"
    echo "⚠️  Warning: Continuing pipeline, but subsequent steps may fail"
    ENHANCING_FAILED=1
fi
echo ""

# Step 4: Detecting
echo "=========================================="
echo "🔄 Step 4: Detecting Agent"
echo "=========================================="
if [ "$ENHANCING_FAILED" = "1" ]; then
    echo "⚠️  Skipping detecting: Enhancing step failed (required input missing)"
    DETECTING_FAILED=1
else
    echo "📊 Running detecting analyst..."
    if python3 agents/detecting/analyst.py "$TUNNEL_ID" > "data/${TUNNEL_ID}/analysis/detecting_analysis.md"; then
        echo "✅ Detecting analysis complete"
    else
        echo "❌ Detecting analyst failed"
        DETECTING_FAILED=1
    fi

    if [ "$DETECTING_FAILED" != "1" ]; then
        echo "💻 Running detecting coder..."
        if python3 agents/detecting/coder.py "$TUNNEL_ID"; then
            echo "✅ Detecting complete"
        else
            echo "❌ Detecting coder failed"
            DETECTING_FAILED=1
        fi
    fi
fi
echo ""

# Step 5: Segmenting
echo "=========================================="
echo "🔄 Step 5: Segmenting Agent"
echo "=========================================="
if [ "$DETECTING_FAILED" = "1" ]; then
    echo "⚠️  Skipping segmenting: Detecting step failed (required input missing)"
    SEGMENTING_FAILED=1
else
    echo "📊 Running segmenting analyst..."
    if python3 agents/segmenting/analyst.py "$TUNNEL_ID" > "data/${TUNNEL_ID}/analysis/segmenting_analysis.md"; then
        echo "✅ Segmenting analysis complete"
    else
        echo "❌ Segmenting analyst failed"
        SEGMENTING_FAILED=1
    fi

    if [ "$SEGMENTING_FAILED" != "1" ]; then
        echo "💻 Running segmenting coder..."
        if python3 agents/segmenting/coder.py "$TUNNEL_ID"; then
            echo "✅ Segmenting complete"
        else
            echo "❌ Segmenting coder failed"
            SEGMENTING_FAILED=1
        fi
    fi
fi
echo ""

# Step 6: Evaluation (optional)
echo "=========================================="
echo "📊 Step 6: Evaluation"
echo "=========================================="
if [ -f "data/${TUNNEL_ID}/only_label.csv" ]; then
    echo "📊 Running evaluation..."
    python3 agents/evaluation.py "$TUNNEL_ID" || {
        echo "⚠️  Evaluation failed (non-critical)"
    }
    echo "✅ Evaluation complete"
else
    echo "⚠️  Skipping evaluation: only_label.csv not found"
fi
echo ""

echo "=========================================="
if [ "$ENHANCING_FAILED" = "1" ] || [ "$DETECTING_FAILED" = "1" ] || [ "$SEGMENTING_FAILED" = "1" ]; then
    echo "⚠️  Pipeline Completed with Warnings for Tunnel: $TUNNEL_ID"
    echo "=========================================="
    echo ""
    echo "Failed steps:"
    [ "$ENHANCING_FAILED" = "1" ] && echo "  ❌ Enhancing"
    [ "$DETECTING_FAILED" = "1" ] && echo "  ❌ Detecting"
    [ "$SEGMENTING_FAILED" = "1" ] && echo "  ❌ Segmenting"
    echo ""
else
    echo "🎉 All Agents Pipeline Complete for Tunnel: $TUNNEL_ID"
    echo "=========================================="
fi
echo ""
echo "📁 Output files:"
echo "  - Parameters: configurable/${TUNNEL_ID}/parameters_*.json"
echo "  - Analysis: data/${TUNNEL_ID}/analysis/*.md"
echo "  - Characteristics: data/ablation/${TUNNEL_ID}/characteristics/*.json"
echo "  - Results: data/${TUNNEL_ID}/*.csv"
if [ -d "data/${TUNNEL_ID}/evaluation" ]; then
    echo "  - Evaluation: data/${TUNNEL_ID}/evaluation/"
fi
echo ""

