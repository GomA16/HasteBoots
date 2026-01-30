#!/bin/bash

# Run the zk_nand_pbs_64 program 10 times, appending each result to the CSV file
# Usage: ./run_10_times.sh

echo "=========================================="
echo "Running SNARK verification 10 times"
echo "Results will be saved to snark_statistics.csv"
echo "=========================================="
echo ""

# Remove the old CSV file (if it exists)
if [ -f "snark_statistics.csv" ]; then
    echo "Removing old snark_statistics.csv..."
    rm snark_statistics.csv
fi

# Run the loop 10 times
for i in {1..10}
do
    echo "=========================================="
    echo "Starting Run #$i"
    echo "=========================================="
    
    # Run the program
    cargo run --release --example zk_nand_zama
    
    # Check exit status
    if [ $? -ne 0 ]; then
        echo "Error: Run #$i failed!"
        exit 1
    fi
    
    echo ""
    echo "Run #$i completed successfully"
    echo ""
done

echo "=========================================="
echo "All 10 runs completed!"
echo "Results saved in: snark_statistics.csv"
echo "=========================================="

# Display summary statistics
if command -v python3 &> /dev/null; then
    echo ""
    echo "Calculating statistics..."
    python3 -c "
import csv

with open('snark_statistics.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

if data:
    prover_times = [float(row['Prover Total (ms)']) for row in data]
    verifier_times = [float(row['Verifier Total (ms)']) for row in data]
    total_sizes = [float(row['Total Size (MB)']) for row in data]
    
    print(f'Average Prover Time: {sum(prover_times)/len(prover_times):.2f} ms')
    print(f'Min Prover Time: {min(prover_times):.2f} ms')
    print(f'Max Prover Time: {max(prover_times):.2f} ms')
    print()
    print(f'Average Verifier Time: {sum(verifier_times)/len(verifier_times):.2f} ms')
    print(f'Min Verifier Time: {min(verifier_times):.2f} ms')
    print(f'Max Verifier Time: {max(verifier_times):.2f} ms')
    print()
    print(f'Average Total Size: {sum(total_sizes)/len(total_sizes):.4f} MB')
"
fi
