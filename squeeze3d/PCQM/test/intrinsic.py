#!/usr/bin/python
# Copyright (C) 2017, 2023, 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import collections
import sys
import getopt
import ConfigParser

org = {}
norm = {}
bitstream = {}
decoded = {}
log = {}
gRange = {}
cRange = {}
pPSNR = {}
numa = {}
numb = {}
seqSet = []
calc = "/data3/tian/quality/pcc_quality/test/pc_error"  # Specify the program of the distortion calculator


def tree():
    return collections.defaultdict(tree)


def getFloatValue(line, keyword):
    valueList = []
    if keyword in line:
        line = line.replace(",", " ")
        for token in line.split():
            try:
                valueList.append(float(token))
            except ValueError:
                continue
    return valueList[0], valueList[1]


def getResults(keyword, logfile):
    ret1 = 0.0
    ret2 = 0.0
    with open(logfile) as inf:
        for line in inf:
            line = line.strip()
            if keyword in line:
                ret1, ret2 = getFloatValue(line, keyword)
    return ret1, ret2


def usage():
    print("./anchor -c config.ini <-r> <-h> <-d data>")


def getSetFromString(str, separator):
    if separator == " ":
        ret = str.split()
    else:
        str = "".join(str.split())
        ret = str.split(separator)
    return ret


def ConfigSectionMap(section):
    numa[section] = ""
    numb[section] = ""
    options = Config.options(section)
    for option in options:
        if option == "org":
            org[section] = Config.get(section, option)
        elif option == "norm":
            norm[section] = Config.get(section, option)
        elif option == "bitstream":
            bitstream[section] = Config.get(section, option)
        elif option == "decoded":
            decoded[section] = Config.get(section, option)
        elif option == "log":
            log[section] = Config.get(section, option)
        elif option == "grange":
            strtmp = Config.get(section, option)
            gRange[section] = map(int, getSetFromString(strtmp, ","))
        elif option == "crange":
            strtmp = Config.get(section, option)
            cRange[section] = map(int, getSetFromString(strtmp, ","))
        elif option == "ppsnr":
            pPSNR[section] = float(Config.get(section, option))
        elif option == "numa":
            numa[section] = int(Config.get(section, option))
        elif option == "numb":
            numb[section] = int(Config.get(section, option))


def main(argv):
    ##########################################
    # Tune this section on what you want to do
    ##########################################
    runCmd = 0  # Set to 1 to run evaluation. Set to 0 to put the Excel sheet ready output and no evaluation would be actually called
    myIni = ""
    seqSetSpecial = ""

    # Update the variables from command line
    try:
        opts, args = getopt.getopt(argv, "hrd:c:", ["help", "run", "data=", "config="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            usage()
            sys.exit()
        elif opt in ("-d", "--data"):
            seqSetSpecial = [arg]
        elif opt in ("-r", "--run"):
            runCmd = 1
        elif opt in ("-c", "--config"):
            myIni = [arg]

    if myIni == "":
        usage()
        sys.exit(2)

    # Load configurations
    Config.read(myIni)
    seqSet = Config.sections()
    # print("sections: %s" % seqSet)

    if seqSetSpecial != "":
        seqSet = seqSetSpecial

    seqSet = sorted(seqSet)

    for seq in seqSet:
        ConfigSectionMap(seq)

    # Do evaluations and reporting
    for seq in seqSet:
        data = tree()
        # Do evaluations
        for g in gRange[seq][0:1]:
            if numa[seq] == "":
                logfile = "%s_intrinsic.txt" % (log[seq])
                cmd = "date; %s -a %s > %s; date" % (calc, org[seq], logfile)
                if runCmd:
                    print("%s" % (cmd))
                    os.system(cmd)
                m1, m2 = getResults("Minimum and maximum NN distances", logfile)
                if runCmd:
                    print("%s -> %f, %f" % (logfile, m1, m2))
                data[g]["min"] = m1
                data[g]["max"] = m2
            else:
                dataLocal1 = []
                dataLocal2 = []
                for frm in range(numa[seq], numb[seq] + 1):
                    logfile = "%s_%d_intrinsic.txt" % (log[seq], frm)
                    fname = "%s%04d.ply" % (org[seq], frm)
                    cmd = "date; %s -a %s > %s; date" % (calc, fname, logfile)
                    if runCmd:
                        print("%s" % (cmd))
                        os.system(cmd)
                    m1, m2 = getResults("Minimum and maximum NN distances", logfile)
                    if runCmd:
                        print("%s -> %f, %f" % (logfile, m1, m2))
                    dataLocal1.append(m1)
                    dataLocal2.append(m2)

                data[g]["min"] = max(dataLocal1)
                data[g]["max"] = max(dataLocal2)

        # Do reporting
        for g in gRange[seq][0:1]:
            print("%s %g" % (seq, data[g]["max"]))


# Init the config variable
Config = ConfigParser.ConfigParser()
if __name__ == "__main__":
    main(sys.argv[1:])
