#!/bin/bash
# Echtzeit-Monitoring für FL Training

LOG_FILE="logs/iid_scaling/32768_clients_run_1.log"

echo "🚀 Federated Learning Progress Monitor"
echo "======================================="
echo "Überwache: $LOG_FILE"
echo ""

# Farben
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

while true; do
    clear
    echo -e "${GREEN}⏰ $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo "======================================="
    
    # 1. Aktuelle Round
    CURRENT_ROUND=$(grep -oP '\[ROUND \K\d+' "$LOG_FILE" | tail -1)
    echo -e "${BLUE}📍 Current Round: ${CURRENT_ROUND:-INIT}${NC} / 75"
    
    # 2. Sampled Clients (wie viele trainieren gerade?)
    SAMPLED=$(grep "sampled" "$LOG_FILE" | tail -1 | grep -oP 'sampled \K\d+')
    echo -e "   Clients training: ${SAMPLED:-0} / 32768"
    
    # 3. Wie viele Clients haben bereits Daten geladen?
    DATA_SPLITS=$(grep -c "Data split:" "$LOG_FILE")
    echo -e "   Clients started: $DATA_SPLITS"
    
    # 4. DatasetHolder Status (sollte nur 1 sein!)
    DATASET_LOADS=$(grep -c "DatasetHolder: Loading dataset" "$LOG_FILE")
    if [ "$DATASET_LOADS" -eq 1 ]; then
        echo -e "${GREEN}   ✅ Dataset loaded ONCE (Ray Shared Memory OK!)${NC}"
    else
        echo -e "${RED}   ⚠️  Dataset loaded ${DATASET_LOADS}x (PROBLEM!)${NC}"
    fi
    
    # 5. RAM Usage
    echo ""
    echo -e "${YELLOW}💾 Server RAM:${NC}"
    free -h | grep Mem | awk '{printf "   Used: %s / %s (%.0f%%)\n", $3, $2, ($3/$2)*100}'
    
    # 6. Ray Actors
    echo ""
    echo -e "${YELLOW}🤖 Ray Processes:${NC}"
    CLIENT_ACTORS=$(ps aux | grep -c "ClientAppActor" || echo 0)
    DATASET_HOLDERS=$(ps aux | grep -c "DatasetHolder" || echo 0)
    echo "   ClientAppActors: $CLIENT_ACTORS"
    echo "   DatasetHolder:   $DATASET_HOLDERS"
    
    # 7. Letzte 5 wichtige Events
    echo ""
    echo -e "${BLUE}📋 Recent Events:${NC}"
    grep -E "\[ROUND|\[INIT|sampled|Best|Evaluation" "$LOG_FILE" | tail -5 | sed 's/^/   /'
    
    # 8. Geschätzte Zeit bis Fertig
    if [ -n "$CURRENT_ROUND" ] && [ "$CURRENT_ROUND" -gt 0 ]; then
        # Berechne durchschnittliche Zeit pro Round
        START_TIME=$(stat -c %Y "$LOG_FILE")
        CURRENT_TIME=$(date +%s)
        ELAPSED=$((CURRENT_TIME - START_TIME))
        
        TIME_PER_ROUND=$((ELAPSED / CURRENT_ROUND))
        REMAINING_ROUNDS=$((75 - CURRENT_ROUND))
        ETA_SECONDS=$((TIME_PER_ROUND * REMAINING_ROUNDS))
        
        ETA_HOURS=$((ETA_SECONDS / 3600))
        ETA_MINS=$(((ETA_SECONDS % 3600) / 60))
        
        echo ""
        echo -e "${YELLOW}⏱️  Timing:${NC}"
        echo "   Time per round: ~$((TIME_PER_ROUND / 60)) min"
        echo "   Estimated completion: ${ETA_HOURS}h ${ETA_MINS}min"
    fi
    
    sleep 10  # Update alle 10 Sekunden
done