## Changes made
- Installed gcc-arm-none-eabi 'apt install gcc-arm-none-eabi'
- Added 'extern' before platform_reset_cause declaration in iotlab-m3.c
- Replaced 'collection.MutableMapping' by 'collections.abc.MutableMapping' in the concerned files (indicated in the raised error)
- Changed site both in exp.sh and the executed command from lille to grenoble (there seems to be a problem with the serial_aggregator at the Lille site)
- Used command for telemetry data generation: ./exp.sh "telemetry" 1 30 20 1 "20,archi=m3:at86rf231+site=grenoble"

### For changing the firmware during an experiment
- Added another source code for the second firmware (broadcast-example-2.c)
- Added it to the target files in Makefile in CODEDIR
- Changed the serial output script from 1> to 1>> to make a concatenation to the serial output
- Added a parameter in the command line ($7) to determine the other number of packets to be sent each time

### For testing the change of configuration
- Added setters to iot-lab/parts/contiki/net/mac/csma.c
- Creating a new firmware test-config.c and adding it to the Makefile
- Generate the object file offline and adding it to the FIT IoT-Lab nodes in the platform