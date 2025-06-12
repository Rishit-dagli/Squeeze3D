#!/usr/bin/python
# Copyright (C) 2017, 2023, 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause

from itertools import izip
import os
import collections
import sys
import getopt
import ConfigParser

# inFile = 'tollbooth_raw.ply'
# normFile = 'tollbooth_normal_ascii_raw.ply'


def main(argv):

    inFile = argv[0]
    normFile = argv[1]
    print("argv: %s, %s" % (inFile, normFile))

    lineNum = 1
    with open(inFile) as inF, open(normFile) as inN:
        for lineF, lineN in izip(inF, inN):
            lineF = lineF.strip()
            lineN = lineN.strip()
            strF = lineF.split()
            strN = lineN.split()
            # if lineNum > 7148500:
            #   # print("Line %d: %s -> %s" % (lineNum, lineF, lineN) )
            #   print("Line %d: %s %s %s %s %s %s %s %s %s %s %s %s %s" % (lineNum, strF[0], strF[1], strF[2], strF[3], strF[4], strF[5], strN[6], strN[7], strN[8], strF[6], strF[7], strF[8], strF[9] ))
            print(
                "%s %s %s %s %s %s %s %s %s %s %s %s %s"
                % (
                    strF[0],
                    strF[1],
                    strF[2],
                    strF[3],
                    strF[4],
                    strF[5],
                    strN[6],
                    strN[7],
                    strN[8],
                    strF[6],
                    strF[7],
                    strF[8],
                    strF[9],
                )
            )
            lineNum = lineNum + 1


if __name__ == "__main__":
    main(sys.argv[1:])
