#!/bin/bash
# exp.sh: launch experiment on IoT-lab, log & retrieve results from server

set -e

#---------------------- TEST ARGUMENTS ----------------------#
if [ "$#" -ne 7 ]; then
	echo "Usage: $0 <exp name> <interval (s)> <payload size (B)> <exp duration (m)> <packets per seconds> <list of nodes>"
	exit
fi
#---------------------- TEST ARGUMENTS ----------------------#

#--------------------- DEFINE VARIABLES ---------------------#
LOGIN="ssimoham"
SITE="grenoble"
IOTLAB="$LOGIN@$SITE.iot-lab.info"
CODEDIR="${HOME}/Desktop/LinkPrediction/iot-lab/parts/contiki/examples/ipv6/simple-udp-rpl"
EXPDIR="${HOME}/Desktop/LinkPrediction/traces-master"
DEFAULT_FIRMWARE_DURATION=$((300*60)) # in seconds
BURSTY_FIRMWARE_DURATION=$((90*60)) # in seconds
#--------------------- DEFINE VARIABLES ---------------------#

#----------------------- CATCH SIGINT -----------------------#
# For a clean exit from the experiment
trap ctrl_c INT
function ctrl_c() {
	echo "Terminating experiment."
	iotlab-experiment stop -i "$EXPID"
	exit 1
}
#----------------------- CATCH SIGINT -----------------------#

#------------------- CONFIGURE FIRMWARE 1 -------------------#
sed -i "s/#define\ SEND_INTERVAL_SECONDS\ .*/#define\ SEND_INTERVAL_SECONDS\ $2/g" $CODEDIR/broadcast-example.c
sed -i "s/#define\ SEND_BUFFER_SIZE\ .*/#define\ SEND_BUFFER_SIZE\ $3/g" $CODEDIR/broadcast-example.c
sed -i "s/#define\ NB_PACKETS\ .*/#define\ NB_PACKETS\ $5/g" $CODEDIR/broadcast-example.c
#------------------- CONFIGURE FIRMWARE 1 -------------------#

#-------------------- COMPILE FIRMWARE 1 --------------------#
cd $CODEDIR
make TARGET=iotlab-m3 -j8 || { echo "Compilation failed."; exit 1; }
#--------------------- COMPILE FIRMWARE ---------------------#

#------------------- CONFIGURE FIRMWARE 2 -------------------#
sed -i "s/#define\ SEND_INTERVAL_SECONDS\ .*/#define\ SEND_INTERVAL_SECONDS\ $2/g" $CODEDIR/broadcast-example-2.c
sed -i "s/#define\ SEND_BUFFER_SIZE\ .*/#define\ SEND_BUFFER_SIZE\ $3/g" $CODEDIR/broadcast-example-2.c
sed -i "s/#define\ NB_PACKETS\ .*/#define\ NB_PACKETS\ $7/g" $CODEDIR/broadcast-example-2.c
#------------------- CONFIGURE FIRMWARE 2 -------------------#

#-------------------- COMPILE FIRMWARE 2 --------------------#
cd $CODEDIR
make TARGET=iotlab-m3 -j8 || { echo "Compilation failed."; exit 1; }
#-------------------- COMPILE FIRMWARE 2 --------------------#

#-------------------- LAUNCH EXPERIMENTS --------------------#
cd $EXPDIR/scripts

# Launch the experiment and obtain its ID
EXPID=$(iotlab-experiment submit -n $1 -d $4 -l $6 | grep id | cut -d' ' -f6)
# Wait for the experiment to begin
iotlab-experiment wait -i $EXPID
# Add a monitoring profile (conso) to the nodes
iotlab-node --update-profile conso -i $EXPID

# Flash nodes with firmware 1 (default)
iotlab-node --flash $CODEDIR/broadcast-example.iotlab-m3 -i $EXPID 
# Wait for contiki
sleep 10 
# Run a script for logging and seeding
iotlab-experiment script -i $EXPID --run $SITE,script=serial_script.sh
sleep $DEFAULT_FIRMWARE_DURATION # Run the default firmware for X s, after that flash the nodes with the second firmware

# Flash nodes with firmware 2 (bursty traffic)
iotlab-node --flash $CODEDIR/broadcast-example-2.iotlab-m3 -i $EXPID 
# Wait for contiki
sleep 10
# Run a script for logging and seeding
iotlab-experiment script -i $EXPID --run $SITE,script=serial_script.sh
sleep $BURSTY_FIRMWARE_DURATION # Run the first firmware for 1 minute, after that flash the nodes with the second firmware

# Flash back nodes with firmware 1 (default)
iotlab-node --flash $CODEDIR/broadcast-example.iotlab-m3 -i $EXPID 
# Wait for contiki
sleep 10
# Run a script for logging and seeding
iotlab-experiment script -i $EXPID --run $SITE,script=serial_script.sh

# Wait for experiment termination
iotlab-experiment wait -i $EXPID --state Terminated
#-------------------- LAUNCH EXPERIMENTS --------------------#


#----------------------- RETRIEVE LOG -----------------------#
ssh $IOTLAB "tar -C ~/.iot-lab/${EXPID}/ -cvzf $1.tar.gz serial_output" 
mkdir $EXPDIR/log/$EXPID
scp "$IOTLAB":~/$1.tar.gz $EXPDIR/log/$EXPID/$1.tar.gz
cd $EXPDIR/log/$EXPID/
tar -xvf $1.tar.gz 
#----------------------- RETRIEVE LOG -----------------------#

exit 0