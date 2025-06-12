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
calc = "/data36/tian/quality/pcc_quality/test/pc_error_0.09a_single_thread"  # Specify the program of the distortion calculator


def prepareJob(jobfile, jobname):
    shFile = "%s.sh" % (jobfile)
    # print("%s" % ( shFile ) )

    file = open(shFile, "w")
    file.write("#!/bin/bash\n")
    file.write("#SBATCH --time=0\n")
    if jobname != "":
        file.write("#SBATCH --job-name=%s\n" % (jobname))
        # file.write( "#SBATCH --partition=grid38\n" % () )
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
    file.write("%s\n" % (cmd))
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
        # file.write( "#SBATCH --partition=grid38\n" % () )
    if "andscape" in cmd or "Stanford" in cmd:
        mem = "14GB"
        file.write("#SBATCH --mem %s\n" % (mem))
    else:
        mem = "5GB"
        file.write("#SBATCH --mem %s\n" % (mem))
    file.write("%s\n" % (cmd))
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
    pPSNR[section] = ""
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
    bSort = 0

    # Update the variables from command line
    try:
        opts, args = getopt.getopt(
            argv, "hrd:c:sb", ["help", "run", "data=", "config=", "sort", "batch"]
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
        elif opt in ("-s", "--sort"):
            bSort = 1

    if myIni == "":
        usage()
        sys.exit(2)

    # Load configurations
    Config.read(myIni)
    seqSet = Config.sections()
    # print("sections: %s" % seqSet)

    if seqSetSpecial != "":
        seqSet = seqSetSpecial
        # print("sections: %s" % seqSet)

    if bSort == 1:
        seqSet = sorted(seqSet)

    for seq in seqSet:
        ConfigSectionMap(seq)

    # Do evaluations and reporting
    for seq in seqSet:
        psnr = tree()
        msee = tree()
        psnrLocal = tree()
        mseeLocal = tree()
        # Do evaluations
        for g in gRange[seq]:
            for c in cRange[seq]:
                if numa[seq] == "":  # Static case
                    decoded = "%s_g%d_c%d.ply" % (decodedPre[seq], g, c)
                    # print("%s" % seq )
                    logfile = "%s_g%d_c%d_psnr.txt" % (log[seq], g, c)
                    jobfile = "%s_g%d_c%d_psnr" % (log[seq], g, c)
                    jobname = "%s%d_%d" % (seq, g, c)

                    bExist = os.path.isfile(decoded)
                    if bExist:
                        if norm[seq]:
                            if pPSNR[seq] == "":
                                cmd = "%s -a %s -b %s -n %s -c         > %s" % (
                                    calc,
                                    org[seq],
                                    decoded,
                                    norm[seq],
                                    logfile,
                                )
                            else:
                                cmd = "%s -a %s -b %s -n %s -c -r %.6g > %s" % (
                                    calc,
                                    org[seq],
                                    decoded,
                                    norm[seq],
                                    pPSNR[seq],
                                    logfile,
                                )
                        else:
                            if pPSNR[seq] == "":
                                cmd = "%s -a %s -b %s -c         > %s" % (
                                    calc,
                                    org[seq],
                                    decoded,
                                    logfile,
                                )
                            else:
                                cmd = "%s -a %s -b %s -c -r %.6g > %s" % (
                                    calc,
                                    org[seq],
                                    decoded,
                                    pPSNR[seq],
                                    logfile,
                                )
                        if runCmd:
                            print("%s" % (cmd))
                            os.system(cmd)
                        if bBatch == 1:
                            submitJob(cmd, jobfile, jobname)
                        else:
                            c2c = getResults("mseF,PSNR (p2point):", logfile)
                            c2p = getResults("mseF,PSNR (p2plane):", logfile)
                            y = getResults("c[0],PSNRF         :", logfile)
                            u = getResults("c[1],PSNRF         :", logfile)
                            v = getResults("c[2],PSNRF         :", logfile)
                            if runCmd:
                                print("%s -> %f, %f" % (logfile, c2c, c2p))
                            psnr[g][c]["c2c"] = c2c
                            psnr[g][c]["c2p"] = c2p
                            psnr[g][c]["y"] = y
                            psnr[g][c]["u"] = u
                            psnr[g][c]["v"] = v
                            c2c = getResults("mseF      (p2point):", logfile)
                            c2p = getResults("mseF      (p2plane):", logfile)
                            y = getResults("c[0],    F         :", logfile)
                            u = getResults("c[1],    F         :", logfile)
                            v = getResults("c[2],    F         :", logfile)
                            msee[g][c]["c2c"] = c2c
                            msee[g][c]["c2p"] = c2p
                            msee[g][c]["y"] = y
                            msee[g][c]["u"] = u
                            msee[g][c]["v"] = v
                    else:
                        print("Not found file %s" % (decoded))
                        return

                else:  # Dynamic case
                    # psnrLocal = tree()
                    # psnrLocal[g][c]['c2c'] = []
                    # psnrLocal[g][c]['c2p'] = []
                    # psnrLocal[g][c]['y'  ] = []
                    # psnrLocal[g][c]['u'  ] = []
                    # psnrLocal[g][c]['v'  ] = []
                    # mseeLocal = tree()
                    # mseeLocal[g][c]['c2c'] = []
                    # mseeLocal[g][c]['c2p'] = []
                    # mseeLocal[g][c]['y'  ] = []
                    # mseeLocal[g][c]['u'  ] = []
                    # mseeLocal[g][c]['v'  ] = []
                    psnrLocal[g][c]["c2c"] = [1.0] * (numb[seq] - numa[seq] + 1)
                    psnrLocal[g][c]["c2p"] = [1.0] * (numb[seq] - numa[seq] + 1)
                    psnrLocal[g][c]["y"] = [1.0] * (numb[seq] - numa[seq] + 1)
                    psnrLocal[g][c]["u"] = [1.0] * (numb[seq] - numa[seq] + 1)
                    psnrLocal[g][c]["v"] = [1.0] * (numb[seq] - numa[seq] + 1)
                    mseeLocal[g][c]["c2c"] = [1.0] * (numb[seq] - numa[seq] + 1)
                    mseeLocal[g][c]["c2p"] = [1.0] * (numb[seq] - numa[seq] + 1)
                    mseeLocal[g][c]["y"] = [1.0] * (numb[seq] - numa[seq] + 1)
                    mseeLocal[g][c]["u"] = [1.0] * (numb[seq] - numa[seq] + 1)
                    mseeLocal[g][c]["v"] = [1.0] * (numb[seq] - numa[seq] + 1)

                    counter = 0
                    idx = 0
                    seqN = seq.replace("AllI", "")
                    for frm in range(numa[seq], numb[seq] + 1):
                        # print("%d  %d" % (counter, frm))
                        fname = "%s%04d.ply" % (org[seq], frm)
                        nname = "%s%04d_n.ply" % (norm[seq], frm)
                        if "AllI" not in seq:
                            decoded = (
                                "%s/%s/%s_GOP2_g%d/%s_GOP2_g%d/%s_%d_g%d_c%d.ply"
                                % (
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
                            )
                        else:
                            decoded = "%s/%s/%s_I_g%d/%s_I_g%d/%s_%d_g%d_c%d.ply" % (
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
                        logfile = "%s_%04d_g%d_c%d_psnr.txt" % (log[seq], frm, g, c)

                        bExist = os.path.isfile(decoded)
                        if bExist:
                            if norm[seq]:
                                if pPSNR[seq] == "":
                                    cmd = "%s -a %s -b %s -n %s -c         > %s" % (
                                        calc,
                                        fname,
                                        decoded,
                                        nname,
                                        logfile,
                                    )
                                else:
                                    cmd = "%s -a %s -b %s -n %s -c -r %.6g > %s" % (
                                        calc,
                                        fname,
                                        decoded,
                                        nname,
                                        pPSNR[seq],
                                        logfile,
                                    )
                            else:
                                if pPSNR[seq] == "":
                                    cmd = "; %s -a %s -b %s -c         > %s" % (
                                        calc,
                                        fname,
                                        decoded,
                                        logfile,
                                    )
                                else:
                                    cmd = "; %s -a %s -b %s -c -r %.6g > %s" % (
                                        calc,
                                        fname,
                                        decoded,
                                        pPSNR[seq],
                                        logfile,
                                    )
                            if runCmd:
                                print("%s" % (cmd))
                                os.system(cmd)

                            if bBatch == 1 and counter == 0:
                                # print("frm=%s" % ( frm ))
                                jobfile = "%s_%d_g%d_c%d_psnr" % (log[seq], frm, g, c)
                                jobname = "%s_%d_%d_%d" % (seq, frm, g, c)
                                prepareJob(jobfile, jobname)
                                addMem(jobfile, "5GB")

                            if bBatch == 1:
                                addJob(jobfile, cmd)
                                if counter == 20 - 1:
                                    finalizeJob(jobfile)
                                    counter = -1  # will be added by 1 at the end
                            else:
                                c2c = getResults("mseF,PSNR (p2point):", logfile)
                                c2p = getResults("mseF,PSNR (p2plane):", logfile)
                                y = getResults("c[0],PSNRF         :", logfile)
                                u = getResults("c[1],PSNRF         :", logfile)
                                v = getResults("c[2],PSNRF         :", logfile)
                                if runCmd:
                                    print("%s -> %f, %f" % (logfile, c2c, c2p))
                                # psnrLocal[g][c]['c2c'].append( c2c )
                                # psnrLocal[g][c]['c2p'].append( c2p )
                                # psnrLocal[g][c]['y'  ].append( y )
                                # psnrLocal[g][c]['u'  ].append( u )
                                # psnrLocal[g][c]['v'  ].append( v )
                                # print(idx)
                                psnrLocal[g][c]["c2c"][idx] = c2c
                                psnrLocal[g][c]["c2p"][idx] = c2p
                                psnrLocal[g][c]["y"][idx] = y
                                psnrLocal[g][c]["u"][idx] = u
                                psnrLocal[g][c]["v"][idx] = v
                                # print('%f' % (psnrLocal[g][c]['c2c'][idx]) )
                                c2c = getResults("mseF      (p2point):", logfile)
                                c2p = getResults("mseF      (p2plane):", logfile)
                                y = getResults("c[0],    F         :", logfile)
                                u = getResults("c[1],    F         :", logfile)
                                v = getResults("c[2],    F         :", logfile)
                                # mseeLocal[g][c]['c2c'].append( c2c )
                                # mseeLocal[g][c]['c2p'].append( c2p )
                                # mseeLocal[g][c]['y'  ].append( y )
                                # mseeLocal[g][c]['u'  ].append( u )
                                # mseeLocal[g][c]['v'  ].append( v )
                                mseeLocal[g][c]["c2c"][idx] = c2c
                                mseeLocal[g][c]["c2p"][idx] = c2p
                                mseeLocal[g][c]["y"][idx] = y
                                mseeLocal[g][c]["u"][idx] = u
                                mseeLocal[g][c]["v"][idx] = v
                        else:
                            print("Not found file %s" % (decoded))
                            return

                        counter = counter + 1
                        idx = idx + 1

                    if bBatch == 1 and counter != 0:
                        print("end frm=%s" % (frm))
                        finalizeJob(jobfile)

                    if bBatch == 0:
                        psnr[g][c]["c2c"] = sum(psnrLocal[g][c]["c2c"]) / len(
                            psnrLocal[g][c]["c2c"]
                        )
                        psnr[g][c]["c2p"] = sum(psnrLocal[g][c]["c2p"]) / len(
                            psnrLocal[g][c]["c2p"]
                        )
                        psnr[g][c]["y"] = sum(psnrLocal[g][c]["y"]) / len(
                            psnrLocal[g][c]["y"]
                        )
                        psnr[g][c]["u"] = sum(psnrLocal[g][c]["u"]) / len(
                            psnrLocal[g][c]["u"]
                        )
                        psnr[g][c]["v"] = sum(psnrLocal[g][c]["v"]) / len(
                            psnrLocal[g][c]["v"]
                        )
                        msee[g][c]["c2c"] = sum(mseeLocal[g][c]["c2c"]) / len(
                            mseeLocal[g][c]["c2c"]
                        )
                        msee[g][c]["c2p"] = sum(mseeLocal[g][c]["c2p"]) / len(
                            mseeLocal[g][c]["c2p"]
                        )
                        msee[g][c]["y"] = sum(mseeLocal[g][c]["y"]) / len(
                            mseeLocal[g][c]["y"]
                        )
                        msee[g][c]["u"] = sum(mseeLocal[g][c]["u"]) / len(
                            mseeLocal[g][c]["u"]
                        )
                        msee[g][c]["v"] = sum(mseeLocal[g][c]["v"]) / len(
                            mseeLocal[g][c]["v"]
                        )

        # Do reporting
        # print('')
        # print('%s point2point point2plane y u v' % (seq))

        if bBatch == 0:
            if numa[seq] == "":  # Static case
                # print('pj = %s' % (pj) )
                for c in cRange[seq]:
                    # print('c = %s' % (c) )
                    for g in gRange[seq]:
                        print(
                            "%s %d  %f  %f  %f  %f  %f  %.6g  %.6g  %.6g  %.6g  %.6g"
                            % (
                                seq,
                                g,
                                psnr[g][c]["c2c"],
                                psnr[g][c]["c2p"],
                                psnr[g][c]["y"],
                                psnr[g][c]["u"],
                                psnr[g][c]["v"],
                                msee[g][c]["c2c"],
                                msee[g][c]["c2p"],
                                msee[g][c]["y"],
                                msee[g][c]["u"],
                                msee[g][c]["v"],
                            )
                        )
            else:  # Dynamic case
                # print('pj = %s' % (pj) )
                for c in cRange[seq]:
                    # print('c = %s' % (c) )
                    for g in gRange[seq]:
                        # print(psnrLocal[g][c])
                        idx = 0
                        for frm in range(numa[seq], numb[seq] + 1):
                            # print('%d' % (idx) )
                            # print('%f' % (psnrLocal[g][c]['c2c'][idx]) )
                            print(
                                "%s_%d %d  %.6g  %.6g  %.6g  %.6g  %.6g  %.6g  %.6g  %.6g  %.6g  %.6g"
                                % (
                                    seq,
                                    frm,
                                    g,
                                    psnrLocal[g][c]["c2c"][idx],
                                    psnrLocal[g][c]["c2p"][idx],
                                    psnrLocal[g][c]["y"][idx],
                                    psnrLocal[g][c]["u"][idx],
                                    psnrLocal[g][c]["v"][idx],
                                    mseeLocal[g][c]["c2c"][idx],
                                    mseeLocal[g][c]["c2p"][idx],
                                    mseeLocal[g][c]["y"][idx],
                                    mseeLocal[g][c]["u"][idx],
                                    mseeLocal[g][c]["v"][idx],
                                )
                            )
                            idx = idx + 1


# Init the config variable
Config = ConfigParser.ConfigParser()
if __name__ == "__main__":
    main(sys.argv[1:])
