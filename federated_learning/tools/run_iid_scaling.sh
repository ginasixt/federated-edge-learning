#!/usr/bin/env bash
# federated_learning/tools/run_iid_scaling_fixed.sh

# Run with:
# ./federated_learning/tools/run_iid_scaling.sh

set -euo pipefail

# IID Scaling: 2 → 177575 Clients (18 Datensätze)
CLIENT_COUNTS=(16384)
RUNS_PER_SPLIT=1

echo "📊 IID SCALING EXPERIMENTS (18 Configurations)"
echo "═══════════════════════════════════════"
echo "Client counts: ${CLIENT_COUNTS[@]}"
echo "Total Training Samples: 177575 (constant)"
echo "Runs per split: ${RUNS_PER_SPLIT}"
echo ""

# 💾 Backup der Original-TOML (nur einmal)
if [ ! -f "pyproject.toml.backup" ]; then
    cp pyproject.toml pyproject.toml.backup
    echo "💾 Created backup: pyproject.toml.backup"
fi

# Cleanup-Funktion
cleanup() {
    echo ""
    echo "🔄 Restoring original pyproject.toml..."
    cp pyproject.toml.backup pyproject.toml
    echo "✅ Original pyproject.toml restored"
    # ✅ NEW
    ray stop --force 2>/dev/null || true
    rm -rf /tmp/ray/
    echo "✅ Ray tmp cleaned"
}

# Cleanup bei Script-Ende (auch bei Fehlern oder Ctrl+C)
trap cleanup EXIT

# Funktion: Update nur options.num-supernodes in TOML
update_supernodes() {
    local num_clients=$1
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/^options\.num-supernodes = .*/options.num-supernodes = ${num_clients}/" pyproject.toml
    else
        # Linux
        sed -i "s/^options\.num-supernodes = .*/options.num-supernodes = ${num_clients}/" pyproject.toml
    fi
    
    echo "   📝 Updated options.num-supernodes = ${num_clients} in pyproject.toml"
}

# Logs vorbereiten
mkdir -p logs/iid_scaling
mkdir -p results/iid_scaling

total_experiments=$((${#CLIENT_COUNTS[@]} * ${RUNS_PER_SPLIT}))
current_experiment=0

for num_clients in "${CLIENT_COUNTS[@]}"; do
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "SCALING: ${num_clients} Clients"
    
    # update options.num-supernodes in TOML, because its not possible over run
    update_supernodes ${num_clients}
    
    # Berechne Samples pro Client
    samples_per_client=$((177575 / num_clients))
    
    echo "   Samples/Client: ~${samples_per_client}"
    echo "════════════════════════════════════════════════════════════════════════════════"
    
    # 🔧 **BEREICH-BESTIMMUNG UND PARAMETER**
    if [ ${num_clients} -lt 100 ]; then
        # **BEREICH 1: 2-64 Clients (Cross-Silo FL)**
        range="Cross-Silo"
        min_fit=$(( (num_clients * 8 + 9) / 10 ))  # 80% für Training
        if [ ${min_fit} -lt 2 ]; then min_fit=2; fi
        min_evaluate=${num_clients}
        rounds=20
        
    elif [ ${num_clients} -lt 1000 ]; then
        # **BEREICH 2: 100-512 Clients (Mid-Scale FL)**  
        range="Mid-Scale"
        min_fit=$(( (num_clients * 6 + 9) / 10 ))   # 60% für Training TODO: nochmal teste wie es mit 80% ausschaut
        if [ ${min_fit} -lt 10 ]; then min_fit=10; fi
        min_evaluate=$([ ${num_clients} -lt 100 ] && echo ${num_clients} || echo 100)
        rounds=30
        
    elif [ ${num_clients} -lt 10000 ]; then
        # **BEREICH 3: 1K-8K Clients (Large-Scale FL)**
        range="Large-Scale"
        min_fit=$(( num_clients * 8 / 10 ))               # 80% für Training
        min_evaluate=$(( num_clients * 6 / 10 ))  # 60% für Evaluation, also 6000 bei 10K
        rounds=50
        
    elif [ ${num_clients} -lt 100000 ]; then
        # **BEREICH 4: 10K-65K Clients (Massive FL)** 
        range="Massive"
        min_fit=$(( num_clients * 6 / 10 ))  # 60% für Training  
        min_evaluate=$(( num_clients * 8 / 10 ))        # 80% für Evaluation
        rounds=80
        
    else
        # **BEREICH 5: 100K+ Clients (Federated Edge Learning)**
        range="FEL"
        min_fit=$(( num_clients / 20 ))              # 5% für Training
        if [ ${min_fit} -lt 500 ]; then min_fit=500; fi
        min_evaluate=$(( num_clients / 4 ))      # 25% für Evaluation
        if [ ${min_evaluate} -lt 2000 ]; then min_evaluate=2000; fi
        rounds=85
    fi
    
    # Alle Bereiche
    min_available=${num_clients}
    split_file="splits_iid_scaling/splits_iid_${num_clients}_clients.json"
    
    echo "📋 Configuration:"
    echo "   Range: ${range}"  
    echo "   options.num-supernodes: ${num_clients} (updated in TOML)"
    echo "   split-path: ${split_file}"
    echo "   min-fit-clients: ${min_fit} ($(( (min_fit * 100) / num_clients ))% participation)"
    echo "   min-available-clients: ${min_available}"
    echo "   min-evaluate-clients: ${min_evaluate}"
    echo "   rounds: ${rounds}"
    echo "   expected-samples/round: $((min_fit * samples_per_client))"
    
    # Prüfe ob Split-File existiert
    if [ ! -f "${split_file}" ]; then
        echo "❌ Split file not found: ${split_file}"
        echo "   Run: python federated_learning/tools/create_iid_scaling_splits.py --min-samples 1"
        continue
    fi
    
    successful_runs=0
    
    for run in $(seq 1 ${RUNS_PER_SPLIT}); do
        current_experiment=$((current_experiment + 1))
        
        echo ""
        echo "[${current_experiment}/${total_experiments}] 🚀 Experiment ${run}/${RUNS_PER_SPLIT}"
        
        log_file="logs/iid_scaling/${num_clients}_clients_run_${run}.log"
        start_time=$(date +%s)
        
        # 🎯 Führe FL-Training aus (num-supernodes bereits in TOML, Rest per --run-config)
       flwr run . --run-config "split-path=\"${split_file}\" min-fit-clients=${min_fit} min-available-clients=${min_available} min-evaluate-clients=${min_evaluate} num-server-rounds=${rounds} run-tag=\"${run}\"" > "${log_file}" 2>&1
        exit_code=$?
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        if [ ${exit_code} -eq 0 ]; then
            echo "   ✅ SUCCESS in ${duration}s (${rounds} rounds)"
            successful_runs=$((successful_runs + 1))
            ray stop --force 2>/dev/null || true
            rm -rf /tmp/ray/
            
            # Kopiere Ergebnisse
            result_pattern="result/*/multi_thr/run_${run}.json"
            for result_file in ${result_pattern}; do
                if [ -f "${result_file}" ]; then
                    cp "${result_file}" "results/iid_scaling/${num_clients}_clients_run_${run}.json"
                    echo "   📊 Saved: results/iid_scaling/${num_clients}_clients_run_${run}.json"
                    break
                fi
            done
        else
            echo "   ❌ FAILED (exit code ${exit_code}) after ${duration}s"
            echo "   📋 Check log: ${log_file}"
        fi
        
        # Pause zwischen Runs
        if [ ${run} -lt ${RUNS_PER_SPLIT} ]; then
            echo "   ⏳ Waiting 3s..."
            ray stop --force 2>/dev/null || true  # ✅ NEW
            rm -rf /tmp/ray/                       # ✅ NEW
            sleep 3
        fi
    done
    
    echo ""
    echo "📊 ${num_clients} Clients: ${successful_runs}/${RUNS_PER_SPLIT} successful"
    
    # ✅ FIXED: Kompatible Array-Zugriff für macOS Bash
    last_client=${CLIENT_COUNTS[${#CLIENT_COUNTS[@]}-1]}
    if [ ${num_clients} -ne ${last_client} ]; then
        echo "   ⏳ Waiting 5s before next scaling step..."
        sleep 5
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "🎉 IID SCALING ANALYSIS COMPLETED"
echo "════════════════════════════════════════════════════════════════════════════════"

# Finale Statistiken
total_results=$(find results/iid_scaling -name "*.json" | wc -l)
echo "📈 Summary:"
echo "   Total experiments: ${total_experiments}"
echo "   Successful: ${total_results}"
echo "   Success rate: $(( (total_results * 100) / total_experiments ))%"
echo ""

echo "📊 Results by Scaling Range:"
echo "   Cross-Silo: $(find results/iid_scaling -name "[0-9]_clients_run_*.json" -o -name "[0-9][0-9]_clients_run_*.json" | wc -l) results"
echo "   Mid-Scale: $(find results/iid_scaling -name "[1-9][0-9][0-9]_clients_run_*.json" | wc -l) results"  
echo "   Large-Scale: $(find results/iid_scaling -name "[1-9][0-9][0-9][0-9]_clients_run_*.json" | wc -l) results"
echo "   Massive: $(find results/iid_scaling -name "[1-6][0-9][0-9][0-9][0-9]_clients_run_*.json" | wc -l) results"
echo "   FEL: $(find results/iid_scaling -name "1[0-9][0-9][0-9][0-9][0-9]_clients_run_*.json" | wc -l) results"

echo ""
echo "📁 Output Locations:"
echo "   Results: results/iid_scaling/*.json" 
echo "   Logs: logs/iid_scaling/*.log"
echo ""
echo "🔄 pyproject.toml will be restored to original state on exit"