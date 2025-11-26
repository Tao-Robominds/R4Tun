#!/bin/bash

# SAM Evolution Pipeline Runner
# This script runs the complete evolution pipeline for all tunnels
# 6-segment tunnels: 1-4, 2-2, 3-1
# 7-segment tunnels: 4-1, 5-1

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

# Function to check if virtual environment is activated
check_venv() {
    if [[ -z "$VIRTUAL_ENV" ]]; then
        print_warning "Virtual environment not detected. Activating venv..."
        source venv/bin/activate
        if [[ $? -ne 0 ]]; then
            print_error "Failed to activate virtual environment. Please run: source venv/bin/activate"
            exit 1
        fi
        print_success "Virtual environment activated"
    else
        print_success "Virtual environment already active: $VIRTUAL_ENV"
    fi
}

# Function to run a command with error handling
run_command() {
    local cmd="$1"
    local description="$2"
    
    print_status "Running: $description"
    echo "Command: $cmd"
    
    if eval "$cmd"; then
        print_success "$description completed successfully"
        return 0
    else
        print_error "$description failed with exit code $?"
        return 1
    fi
}

# Function to run evolution pipeline for 6-segment tunnels
run_6_segment_pipeline() {
    local tunnel_id="$1"
    local base_dir="ablation/6.self_reflecting"
    
    echo ""
    echo "========================================================"
    print_step "RUNNING 6-SEGMENT PIPELINE FOR TUNNEL $tunnel_id"
    echo "========================================================"
    
    # Step 1: Run SAM Evolution
    print_step "Step 1/3: Running SAM Parameter Evolution"
    if ! run_command "python $base_dir/sam_evolver.py $tunnel_id" "SAM Evolution for $tunnel_id"; then
        print_warning "SAM Evolution failed for $tunnel_id, but continuing..."
    fi
    
    # Step 2: Run SAM Segmentation
    print_step "Step 2/3: Running SAM Segmentation"
    if ! run_command "python $base_dir/sam.py $tunnel_id" "SAM Segmentation for $tunnel_id"; then
        print_error "SAM Segmentation failed for $tunnel_id. Skipping evaluation."
        return 1
    fi
    
    # Step 3: Run Evaluation
    print_step "Step 3/3: Running Performance Evaluation"
    if ! run_command "python $base_dir/evaluation.py $tunnel_id" "Performance Evaluation for $tunnel_id"; then
        print_warning "Evaluation failed for $tunnel_id"
        return 1
    fi
    
    print_success "6-segment pipeline completed for tunnel $tunnel_id"
    return 0
}

# Function to run evolution pipeline for 7-segment tunnels
run_7_segment_pipeline() {
    local tunnel_id="$1"
    local base_dir="ablation/6.self_reflecting"
    
    echo ""
    echo "========================================================"
    print_step "RUNNING 7-SEGMENT PIPELINE FOR TUNNEL $tunnel_id"
    echo "========================================================"
    
    # Step 1: Run SAM Evolution (4+5 version)
    print_step "Step 1/3: Running SAM Parameter Evolution (7-segment)"
    if ! run_command "python $base_dir/sam_evolver_4+5.py $tunnel_id" "SAM Evolution (4+5) for $tunnel_id"; then
        print_warning "SAM Evolution failed for $tunnel_id, but continuing..."
    fi
    
    # Step 2: Run SAM Segmentation (4+5 version)
    print_step "Step 2/3: Running SAM Segmentation (7-segment)"
    if ! run_command "python $base_dir/sam_4+5.py $tunnel_id" "SAM Segmentation (4+5) for $tunnel_id"; then
        print_error "SAM Segmentation failed for $tunnel_id. Skipping evaluation."
        return 1
    fi
    
    # Step 3: Run Evaluation (4+5 version)
    print_step "Step 3/3: Running Performance Evaluation (7-segment)"
    if ! run_command "python $base_dir/evaluation_4+5.py $tunnel_id" "Performance Evaluation (4+5) for $tunnel_id"; then
        print_warning "Evaluation failed for $tunnel_id"
        return 1
    fi
    
    print_success "7-segment pipeline completed for tunnel $tunnel_id"
    return 0
}

# Function to check if required files exist for a tunnel
check_tunnel_requirements() {
    local tunnel_id="$1"
    local required_files=("detected.csv" "pixel_to_point.pkl" "enhanced.csv" "ring_count.txt" "depth_map.png")
    
    print_status "Checking requirements for tunnel $tunnel_id..."
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "data/$tunnel_id/$file" ]]; then
            print_error "Required file missing: data/$tunnel_id/$file"
            return 1
        fi
    done
    
    print_success "All required files found for tunnel $tunnel_id"
    return 0
}

# Function to display summary
display_summary() {
    local success_count="$1"
    local total_count="$2"
    
    echo ""
    echo "========================================================"
    print_step "PIPELINE EXECUTION SUMMARY"
    echo "========================================================"
    
    if [[ $success_count -eq $total_count ]]; then
        print_success "All $total_count tunnels processed successfully!"
    else
        print_warning "$success_count out of $total_count tunnels processed successfully"
        print_warning "$((total_count - success_count)) tunnels had issues"
    fi
    
    echo ""
    print_status "Results can be found in:"
    print_status "- SAM evolution analysis: data/{tunnel_id}/analysis/"
    print_status "- Performance metrics: data/{tunnel_id}/evaluation/"
    print_status "- Segmentation results: data/{tunnel_id}/final.csv"
}

# Main execution
main() {
    echo "========================================================"
    echo "        SAM EVOLUTION PIPELINE RUNNER"
    echo "========================================================"
    echo "This script will run the complete evolution pipeline:"
    echo "1. Parameter Evolution → 2. SAM Segmentation → 3. Evaluation"
    echo ""
    echo "6-segment tunnels: 1-4, 2-2, 3-1"
    echo "7-segment tunnels: 4-1, 5-1"
    echo "========================================================"
    
    # Check and activate virtual environment
    check_venv
    
    # Change to project directory
    if [[ ! -d "ablation/6.self_reflecting" ]]; then
        print_error "Project directory not found. Please run from the R4Tun root directory."
        exit 1
    fi
    
    # Define tunnel arrays
    six_segment_tunnels=("1-4" "2-2" "3-1")
    seven_segment_tunnels=("4-1" "5-1")
    
    local success_count=0
    local total_count=5
    local failed_tunnels=()
    
    # Process 6-segment tunnels
    print_step "PROCESSING 6-SEGMENT TUNNELS"
    for tunnel in "${six_segment_tunnels[@]}"; do
        if check_tunnel_requirements "$tunnel"; then
            if run_6_segment_pipeline "$tunnel"; then
                ((success_count++))
            else
                failed_tunnels+=("$tunnel (6-segment)")
            fi
        else
            print_error "Skipping tunnel $tunnel due to missing requirements"
            failed_tunnels+=("$tunnel (6-segment - missing files)")
        fi
        
        # Add separator between tunnels
        echo ""
        sleep 1
    done
    
    # Process 7-segment tunnels
    print_step "PROCESSING 7-SEGMENT TUNNELS"
    for tunnel in "${seven_segment_tunnels[@]}"; do
        if check_tunnel_requirements "$tunnel"; then
            if run_7_segment_pipeline "$tunnel"; then
                ((success_count++))
            else
                failed_tunnels+=("$tunnel (7-segment)")
            fi
        else
            print_error "Skipping tunnel $tunnel due to missing requirements"
            failed_tunnels+=("$tunnel (7-segment - missing files)")
        fi
        
        # Add separator between tunnels
        echo ""
        sleep 1
    done
    
    # Display summary
    display_summary "$success_count" "$total_count"
    
    # Display failed tunnels if any
    if [[ ${#failed_tunnels[@]} -gt 0 ]]; then
        echo ""
        print_warning "Failed tunnels:"
        for failed in "${failed_tunnels[@]}"; do
            echo "  - $failed"
        done
    fi
    
    echo ""
    print_status "Pipeline execution completed."
    
    # Return appropriate exit code
    if [[ $success_count -eq $total_count ]]; then
        exit 0
    else
        exit 1
    fi
}

# Check if script is being run with arguments for individual tunnel processing
if [[ $# -eq 1 ]]; then
    tunnel_id="$1"
    
    check_venv
    
    # Determine tunnel type and run appropriate pipeline
    if [[ "$tunnel_id" == "1-4" || "$tunnel_id" == "2-2" || "$tunnel_id" == "3-1" ]]; then
        print_status "Running 6-segment pipeline for tunnel $tunnel_id"
        if check_tunnel_requirements "$tunnel_id"; then
            run_6_segment_pipeline "$tunnel_id"
        else
            print_error "Missing requirements for tunnel $tunnel_id"
            exit 1
        fi
    elif [[ "$tunnel_id" == "4-1" || "$tunnel_id" == "5-1" ]]; then
        print_status "Running 7-segment pipeline for tunnel $tunnel_id"
        if check_tunnel_requirements "$tunnel_id"; then
            run_7_segment_pipeline "$tunnel_id"
        else
            print_error "Missing requirements for tunnel $tunnel_id"
            exit 1
        fi
    else
        print_error "Invalid tunnel ID: $tunnel_id"
        print_status "Valid tunnel IDs: 1-4, 2-2, 3-1, 4-1, 5-1"
        exit 1
    fi
elif [[ $# -eq 0 ]]; then
    # Run full pipeline
    main
else
    print_error "Usage: $0 [tunnel_id]"
    print_status "Examples:"
    print_status "  $0           # Run all tunnels"
    print_status "  $0 1-4       # Run only tunnel 1-4"
    print_status "  $0 4-1       # Run only tunnel 4-1"
    exit 1
fi
