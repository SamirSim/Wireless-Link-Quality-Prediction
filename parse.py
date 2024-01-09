#!/usr/bin/python3
#
#   Copyright 2021 FOUBERT Brandon
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
 

import os
import sys
import matplotlib.pyplot as plt

from shutil import copyfile
#from nodes import Node, MessageTx, MessageRx
#from nodes import averageTransmissionLatency, packetDeliveryRatio


def parseLog(log_filename):
    """Take a log file an return a list of Node found in it 

    Parameters
    ----------
    log_filename 
        the name of the file containing the experiment's log
    
    Returns
    -------
    list
        a list of Nodes from the log file
    """

    try:
       log_file = open(log_filename, "r" )
    except IOError:
        print(sys.argv[0]+": "+log_filename+": cannot open file")
        sys.exit(3)

    lines = log_file.readlines()
    nodes_dict = {}
    for line in lines:
        receiver = "Received" in line
        sender = "Sending" in line
        if not sender and not receiver :
            continue
        try:
            split = line.split(';')
            timestamp = float(split[0])*1000 # Convert in milliseconds
            name = split[1]
            msg = split[2]
            try : 
                msg_id = int(split[3])
            except : 
                msg_id = int(split[3], 16)

            # If we find a new node
            if not name in nodes_dict:
                # We create a new Node in the nodes dict
                nodes_dict[name] = Node(name)

            if receiver :
                nodes_dict[name].list_rx.append(MessageRx(timestamp, msg_id, name))
            # If sender 
            else :
                nodes_dict[name].list_tx.append(MessageTx(timestamp, msg_id, name))

        except IndexError as e:
            print("Bad formatted line : {}".format(e))

    nodes_list = nodes_dict.values()
    return nodes_list


def beautifyData(exp_name, nodes_list):
    """Write in file the raw data in a more readable way
    It will write all the transmitted messages in a filenameRx.data
    and received messages in a filenameTx.data

    The two file are store in a directory with the experience's name
    
    Format of the Rx.data :
        transmitted;<node name>;<timestamp>;<message id>;<transmission success>

    Format of the Tx.data :
        received;<node name>;<timestamp>;<message id>;<reception delay>

    where
        <transmission success> is the nb of node that have received this message
        <reception delay> is the time that the message has taken to reach the node in milliseconds

    Parameters
    ----------
    exp_name 
        the experience's name used to name and store the beautified data 
    nodes_list
        a list of Node

    Notes
    -----
    
    """
    if not os.path.exists(exp_name):
        os.makedirs(exp_name)

    rx_name = "{}/{}Rx.data".format(exp_name, exp_name)
    tx_name = "{}/{}Tx.data".format(exp_name, exp_name)

    with open(rx_name, "w") as rx_file, open(tx_name, "w") as tx_file: 
        tx_file.write("transmitted;node name;timestamp;message id;transmission success\n")
        rx_file.write("received;node name;timestamp;message id;reception delay\n")
        for n in nodes_list:
            for t in n.list_tx:
                tx_file.write('{}\n'.format(t))
            for r in n.list_rx:
                rx_file.write('{}\n'.format(r))

def generatePlots(exp_name, nodes_list):
    """Generate plots from the list of nodes

    Generate a graph for the packet delivery ratio of each nodes
    and a graph for the average transmission latency of each nodes in miliseconds

    Parameters
    ----------
    exp_name 
        the name of the experience, used to store the plot
        in the corresponding directory
    nodes_list
        a list of Node

    Notes
    -----
    Packet delivery ratio = ∑ Number of packet received / ∑ Number of packet send
    
    """

    if not os.path.exists(exp_name):
        os.makedirs(exp_name)

    names = []
    pdr = []
    latency = []

    plt.rcdefaults()

    for n in nodes_list:
        names.append(n.name)
        pdr.append(n.pdr)
        latency.append(n.latency)

    # Plot pdr 
    fig = plt.figure(figsize=(15, 15))
    plt.xticks(rotation='vertical')
    plt.bar(names, pdr, width=0.3)
    plt.xlabel("Node ID")
    plt.ylabel("PDR (%)")
    plt.title("Packet Delivery Ratio (PDR) per node")
    #plt.show()
    plt.savefig(exp_name + "/PacketDeliveryRatio.png")


    # Plot latency 
    fig = plt.figure(figsize=(14, 15))
    plt.xticks(rotation='vertical')
    plt.bar(names, latency, align="edge", width=0.3)
    plt.xlabel("Node ID")
    plt.ylabel("Average Latency (ms)")
    plt.title("Average packet transmission latency per node")
    #plt.show()
    plt.savefig(exp_name + "/AverageTransmissionLatency.png")


if __name__ == "__main__":

    if len(sys.argv) != 3 :
        print("Usage: "+sys.argv[0]+" <log filename> <exp name>")
        sys.exit(2)
    log_filename = sys.argv[1]
    exp_name = sys.argv[2]

    nodes_list = parseLog(log_filename)
    #packetDeliveryRatio(nodes_list)
    #averageTransmissionLatency(nodes_list)

    beautifyData(exp_name, nodes_list)
    generatePlots(exp_name, nodes_list)

    # Just copy the rawdata in the same directory as beautified data
    if not os.path.exists(exp_name):
        os.makedirs(exp_name)
    copyfile(log_filename, "{}/{}.rawdata".format(exp_name, exp_name))