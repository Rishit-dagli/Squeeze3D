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
bitstreamPre = {}
decodedPre = {}
log = {}
gRange = {}
cRange = {}
pPSNR = {}
numa = {}
numb = {}
seqSet = []
# decoderFrm = '/data3/tian/anchor_codec/pcc-mp3dg-release/bin/pcl_mpeg_pcc_frame_coder'  # Specify the program of the distortion calculator
# decoderGop = '/data3/tian/anchor_codec/pcc-mp3dg-release/bin/pcl_mpeg_pcc_gop_coder'  # Specify the program of the distortion calculator
decoderFrm = "/data36/tian/anchor_codec/pcc-mp3dg_after118.release/bin/pcl_mpeg_pcc_frame_coder"  # Specify the program of the distortion calculator
decoderGop = "/data36/tian/anchor_codec/pcc-mp3dg_after118.release/bin/pcl_mpeg_pcc_gop_coder"  # Specify the program of the distortion calculator


def prepareJob(jobfile, jobname):
    shFile = "%s.sh" % (jobfile)
    # print("%s" % ( shFile ) )

    file = open(shFile, "w")
    file.write("#!/bin/bash\n")
    file.write("#SBATCH --time=0\n")
    if jobname != "":
        file.write("#SBATCH --job-name=%s\n" % (jobname))
    file.close()


def addMem(jobfile, mem):
    shFile = "%s.sh" % (jobfile)
    # print("%s" % ( shFile ) )

    file = open(shFile, "a")
    file.write("#SBATCH --mem %s\n" % (mem))
    file.close()


def addJob(jobfile, cmd):
    shFile = "%s.sh" % (jobfile)
    # print("%s" % ( shFile ) )

    file = open(shFile, "a")
    # file.write("srun -l %s\n" %( "sleep 20" ) )
    file.write("%s\n" % (cmd))  # srun -l
    file.close()


def finalizeJob(jobfile):
    shFile = "%s.sh" % (jobfile)
    shLog = "%s_slurm.log" % (jobfile)
    print("%s" % (shFile))

    myJob = "sbatch -o %s %s" % (shLog, shFile)
    os.system(myJob)


def submitJob(cmd, jobfile, jobname):
    shFile = "%s.sh" % (jobfile)
    shLog = "%s_slurm.log" % (jobfile)
    print("%s" % (shFile))

    file = open(shFile, "w")
    file.write("#!/bin/bash\n")
    file.write("#SBATCH --time=0\n")
    if jobname != "":
        file.write("#SBATCH --job-name=%s\n" % (jobname))
    if "andscape" in cmd or "Stanford" in cmd:
        mem = "14GB"
        file.write("#SBATCH --mem %s\n" % (mem))
    else:
        mem = "5GB"
        file.write("#SBATCH --mem %s\n" % (mem))
    file.write("%s\n" % (cmd))
    # file.write("srun -l %s\n" %( "sleep 20" ) )
    file.close()

    myJob = "sbatch -o %s %s" % (shLog, shFile)
    os.system(myJob)


def tree():
    return collections.defaultdict(tree)


def getFloatValue(line, keyword):
    flag = 0
    value = 0.0
    if keyword in line:
        for token in line.split():
            try:
                if float(token):
                    flag = 1
                    value = float(token)
            except ValueError:
                continue
    return flag, value


def getResults(keyword, logfile):
    ret = 0.0
    with open(logfile) as inf:
        for line in inf:
            line = line.strip()
            if keyword in line:
                flag, value = getFloatValue(line, keyword)
                if flag:
                    ret = value
    return ret


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
            bitstreamPre[section] = Config.get(section, option)
        elif option == "decoded":
            decodedPre[section] = Config.get(section, option)
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
    bBatch = 0

    # Update the variables from command line
    try:
        opts, args = getopt.getopt(
            argv, "hrbd:c:", ["help", "run", "batch", "data=", "config="]
        )
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
        elif opt in ("-b", "--batch"):
            bBatch = 1

    if myIni == "":
        usage()
        sys.exit(2)

    # Load configurations
    Config.read(myIni)
    seqSet = Config.sections()
    # print("sections: %s" % seqSet)

    if seqSetSpecial != "":
        seqSet = seqSetSpecial

    for seq in seqSet:
        ConfigSectionMap(seq)

    # Do evaluations and reporting
    for seq in seqSet:
        data = tree()
        # Do evaluations
        for g in gRange[seq]:
            for c in cRange[seq]:
                if numa[seq] == "":  # Static case
                    bitstream = "%s_g%d_c%d.pcc" % (bitstreamPre[seq], g, c)
                    decoded = "%s_g%d_c%d.ply" % (decodedPre[seq], g, c)
                    logfile = "%s_g%d_c%d_decode.txt" % (log[seq], g, c)
                    jobfile = "%s_g%d_c%d_decode" % (log[seq], g, c)
                    jobname = "%s_%d_%d" % (seq, g, c)

                    bExist = os.path.isfile(bitstream)
                    if bExist:
                        cmd = "%s -d %s %s > %s" % (
                            decoderFrm,
                            decoded,
                            bitstream,
                            logfile,
                        )
                        if runCmd:
                            print("%s" % (cmd))
                            os.system(cmd)
                        if bBatch:
                            submitJob(cmd, jobfile, jobname)
                    else:
                        print("Not found file %s" % (bitstream))
                        return

                else:  # Dynamic case

                    counter = 0
                    if "AllI" not in seq:  # GOP2 case
                        frm = numa[seq]
                        while frm <= numb[seq]:
                            if "queen" in seq:
                                bitstream = (
                                    "%s/%s/%s_GOP2_g%d/%s_GOP2_g%d/%s_%04d_%04d_g%d_c%d.pcc"
                                    % (
                                        bitstreamPre[seq],
                                        seq,
                                        seq,
                                        g,
                                        seq,
                                        g,
                                        seq,
                                        frm,
                                        frm + 1,
                                        g,
                                        c,
                                    )
                                )
                            elif (
                                "longdress" in seq
                                or "loot" in seq
                                or "redandblack" in seq
                                or "soldier" in seq
                            ):  # 8i has string "vox10"
                                bitstream = (
                                    "%s/%s/%s_GOP2_g%d_anchors/%s_GOP2_g%d/%s_%04d_%04d_g%d_c%d.pcc"
                                    % (
                                        bitstreamPre[seq],
                                        seq,
                                        seq,
                                        g,
                                        seq,
                                        g,
                                        seq,
                                        frm,
                                        frm + 1,
                                        g,
                                        c,
                                    )
                                )
                            else:  # ford
                                bitstream = (
                                    "%s/%s/%s_GOP2_g%d_anchors/%s_GOP2_g%d/%s_%04d_%04d_g%d_c%d.pcc"
                                    % (
                                        bitstreamPre[seq],
                                        seq,
                                        seq,
                                        g,
                                        seq,
                                        g,
                                        seq,
                                        frm,
                                        frm + 1,
                                        g,
                                        c,
                                    )
                                )
                            decodedI = (
                                "%s/%s/%s_GOP2_g%d/%s_GOP2_g%d/%s_%d_g%d_c%d.ply"
                                % (decodedPre[seq], seq, seq, g, seq, g, seq, frm, g, c)
                            )
                            decodedP = (
                                "%s/%s/%s_GOP2_g%d/%s_GOP2_g%d/%s_%d_g%d_c%d.ply"
                                % (
                                    decodedPre[seq],
                                    seq,
                                    seq,
                                    g,
                                    seq,
                                    g,
                                    seq,
                                    frm + 1,
                                    g,
                                    c,
                                )
                            )
                            logfile = "%s_%04d_g%d_c%d_decode.txt" % (
                                log[seq],
                                frm,
                                g,
                                c,
                            )
                            bExist = os.path.isfile(bitstream)
                            if bExist:
                                cmd = "%s -d %s %s %s > %s" % (
                                    decoderGop,
                                    decodedI,
                                    decodedP,
                                    bitstream,
                                    logfile,
                                )
                                if runCmd:
                                    print("%s" % (cmd))
                                    os.system(cmd)
                                if bBatch == 1 and counter == 0:
                                    # print("frm=%s" % ( frm ))
                                    jobfile = "%s_%d_g%d_c%d_decode" % (
                                        log[seq],
                                        frm,
                                        g,
                                        c,
                                    )
                                    jobname = "%s_%d_%d_%d" % (seq, frm, g, c)
                                    prepareJob(jobfile, jobname)
                                    addMem(jobfile, "5GB")
                                if bBatch == 1:
                                    addJob(jobfile, cmd)
                                    if counter >= 20 - 2:
                                        finalizeJob(jobfile)
                                        counter = -2  # will be added by 2 at the end

                            else:
                                print("Not found file %s" % (bitstream))
                                return
                            frm += 2
                            counter += 2

                        if bBatch == 1 and counter != 0:
                            print("end frm=%s" % (frm))
                            finalizeJob(jobfile)

                    else:  # All I case
                        frm = numa[seq]
                        seqN = seq.replace("AllI", "")
                        while frm <= numb[seq]:
                            if (
                                "longdress" in seq
                                or "loot" in seq
                                or "redandblack" in seq
                                or "soldier" in seq
                            ):  # 8i has string "vox10"
                                bitstream = (
                                    "%s/%s/%s_I_g%d_anchors/%s_I_g%d/%s_vox10_%04d_g%d_c%d.pcc"
                                    % (
                                        bitstreamPre[seq],
                                        seqN,
                                        seqN,
                                        g,
                                        seqN,
                                        g,
                                        seqN,
                                        frm,
                                        g,
                                        c,
                                    )
                                )
                            elif "queen" in seq:
                                bitstream = (
                                    "%s/%s/%s_I_g%d/%s_I_g%d/frame_%04d_g%d_c%d.pcc"
                                    % (
                                        bitstreamPre[seq],
                                        seqN,
                                        seqN,
                                        g,
                                        seqN,
                                        g,
                                        frm,
                                        g,
                                        c,
                                    )
                                )
                            else:  # Ford now.
                                bitstream = (
                                    "%s/%s/%s_I_g%d_anchors/%s_I_g%d/%s-%04d_g%d_c%d.pcc"
                                    % (
                                        bitstreamPre[seq],
                                        seqN,
                                        seqN,
                                        g,
                                        seqN,
                                        g,
                                        seqN,
                                        frm,
                                        g,
                                        c,
                                    )
                                )
                            decodedI = "%s/%s/%s_I_g%d/%s_I_g%d/%s_%d_g%d_c%d.ply" % (
                                decodedPre[seq],
                                seqN,
                                seqN,
                                g,
                                seqN,
                                g,
                                seqN,
                                frm,
                                g,
                                c,
                            )
                            logfile = "%s_%04d_g%d_c%d_decode.txt" % (
                                log[seq],
                                frm,
                                g,
                                c,
                            )
                            bExist = os.path.isfile(bitstream)
                            if bExist:
                                cmd = "date; %s -d %s %s > %s; date" % (
                                    decoderFrm,
                                    decodedI,
                                    bitstream,
                                    logfile,
                                )
                                if runCmd:
                                    print("%s" % (cmd))
                                    print("hahaha")
                                    os.system(cmd)
                                if bBatch == 1 and counter == 0:
                                    # print("frm=%s" % ( frm ))
                                    jobfile = "%s_%d_g%d_c%d_decode" % (
                                        log[seq],
                                        frm,
                                        g,
                                        c,
                                    )
                                    jobname = "%s_%d_%d_%d" % (seq, frm, g, c)
                                    prepareJob(jobfile, jobname)
                                    addMem(jobfile, "5GB")
                                if bBatch == 1:
                                    addJob(jobfile, cmd)
                                    if counter == 20 - 1:
                                        finalizeJob(jobfile)
                                        counter = -1  # will be added by 1 at the end
                            else:
                                print("Not found file %s" % (bitstream))
                                return
                            frm += 1
                            counter += 1

                        if bBatch == 1 and counter != 0:
                            print("end frm=%s" % (frm))
                            finalizeJob(jobfile)

        # Do reporting
        print("")
        print("%s " % (seq))


# Init the config variable
Config = ConfigParser.ConfigParser()
if __name__ == "__main__":
    main(sys.argv[1:])
