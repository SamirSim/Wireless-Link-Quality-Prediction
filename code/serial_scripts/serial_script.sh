#!/bin/bash
# Script from : https://www.iot-lab.info/docs/tools/run-script/

# Redirect all nodes serial output to a file
readonly OUTFILE="${HOME}/.iot-lab/${EXP_ID}/serial_output"

# Get nodes list 
NODES=$(iotlab-experiment get -d -i $EXP_ID | jq '.[]' | jq @sh)

# Send nodes a init seed 
for i in $NODES
do
    n=$(tr -d "'\"" <<< $i)
    socat -T1 - TCP:$n:20000 <<< $RANDOM
done
sleep 1
# Send nodes a seq seed 
for i in $NODES
do
    n=$(tr -d "'\"" <<< $i)
    socat -T1 - TCP:$n:20000 <<< $RANDOM
done
sleep 1

serial_aggregator -i ${EXP_ID} 2> /dev/null 1>> ${OUTFILE}
