// Copyright (C) 2017, 2023, 2025 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <sstream>
#include <boost/program_options.hpp>
#include "pcc_processing.hpp"
#include "pcc_distortion.hpp"
#include "clockcom.hpp"

using namespace std;
using namespace boost::program_options;
using namespace pcc_quality;
using namespace pcc_processing;

void printusage()
{
  cout << "pc_psnr cloud_a cloud_b [radiusTimes]" << endl;
  cout << "  default radiusTimes is 10" << endl;
}

int parseCommand( int ac, char * av[], commandPar &cPar )
{
  try {
    options_description desc("Allowed options");
    desc.add_options()
      ("help,h", "produce help message")
      ("fileA,a", value(&cPar.file1)->required(), "Input file 1, original version") // value< string >
      ("fileB,b", value(&cPar.file2)->default_value(""), "Input file 2, processed version")
      ("inputNorm,n", value(&cPar.normIn)->default_value(""), "File name to import the normals of original point cloud, if different from original file 1")
      ("singlePass,s", bool_switch(&cPar.singlePass)->default_value(false), "Force running a single pass, where the loop is over the original point cloud")
      ("hausdorff,d", bool_switch(&cPar.hausdorff)->default_value(false), "Send the Haursdorff metric as well")
      ("color,c", bool_switch(&cPar.bColor)->default_value(false), "Check color distortion as well")
      ("lidar,l", bool_switch(&cPar.bLidar)->default_value(false), "Check lidar reflectance as well")
      ("resolution,r", value(&cPar.resolution)->default_value(0), "Specify the intrinsic resolution")
      ;

    // positional_options_description p;
    // p.add("rtimes", -1);
    variables_map vm;
    store(parse_command_line(ac, av, desc), vm);

    if (ac == 1 || vm.count("help")) { // @DT: Important too add ac == 1
      cout << "Usage: " << av[0] << " [options]\n";
      cout << desc;
      return 0;
    }

    notify(vm);                 // @DT: Report any missing parameters

    // It is wierd the variables were not set. Force the job
    cPar.file1 = vm["fileA"].as< string >();
    cPar.file2 = vm["fileB"].as< string >();
    cPar.normIn = vm["inputNorm"].as< string >();
    cPar.singlePass = vm["singlePass"].as< bool >();
    cPar.bColor = vm["color"].as< bool >();
    cPar.bLidar = vm["lidar"].as< bool >();
    cPar.resolution = vm["resolution"].as< float >();

    if (cPar.normIn == "")
      cPar.c2c_only = true;
    else
      cPar.c2c_only = false;
    // Safety check

    // Check whether your system is compatible with my assumptions
    int szFloat = sizeof(float)*8;
    int szDouble = sizeof(double)*8;
    int szShort = sizeof(short)*8;
    int szInt = sizeof(int)*8;
    // int szLong = sizeof(long long)*8;

    if ( szFloat != 32 || szDouble != 64 || szShort != 16 || szInt != 32 ) //  || szLong != 64
    {
      cout << "Warning: Your system is incompatible with our assumptions below: " << endl;
      cout << "float: "<< sizeof(float)*8 << endl;
      cout << "double: "<< sizeof(double)*8 << endl;
      cout << "short: "<< sizeof(short)*8 << endl;
      cout << "int: "<< sizeof(int)*8 << endl;
      // cout << "long long: "<< sizeof(long long)*8 << endl;
      cout << endl;
      return 0;
    }

    return 1;
  }

  catch(std::exception& e)
  {
    cout << e.what() << "\n";
    return 0;
  }

  // Confict check

}

void printCommand( commandPar &cPar )
{
  cout << "infile1: " << cPar.file1 << endl;
  cout << "infile2: " << cPar.file2 << endl;
  cout << "normal1: " << cPar.normIn << endl;

  if (cPar.singlePass)
    cout << "force running a single pass" << endl;

  cout << endl;
}

int main (int argc, char *argv[])
{
  // Print the version information
  cout << "PCC quality measurement software, version " << PCC_QUALITY_VERSION << endl << endl;

  commandPar cPar;
  if ( parseCommand( argc, argv, cPar ) == 0 )
    return 0;

  printCommand( cPar );

  PccPointCloud inCloud1;
  PccPointCloud inCloud2;
  PccPointCloud inNormal1;

  if (inCloud1.load(cPar.file1))
  {
    cout << "Error reading reference point cloud:" << cPar.file1 << endl;
    return -1;
  }
  cout << "Reading file 1 done." << endl;

  if (cPar.normIn != "")
  {
    if (inNormal1.load(cPar.normIn, true))
    {
      cout << "Error reading normal reference point cloud:" << cPar.normIn << endl;
      return -1;
    }
    cout << "Reading normal 1 done." << endl;
  }

  if (cPar.file2 != "")
  {
    if (inCloud2.load(cPar.file2))
    {
      cout << "Error reading the second point cloud: " << cPar.file2 << endl;
      return -1;
    }
    cout << "Reading file 2 done." << endl;
  }

  // compute the point to plane distances, as well as point to point distances
  const int t0 = GetTickCount();
  qMetric qm;
  computeQualityMetric(inCloud1, inNormal1, inCloud2, cPar, qm);

  const int t1 = GetTickCount();
  cout << "Job done! " << (t1 - t0) * 1e-3 << " seconds elapsed (excluding the time to load the point clouds)." << endl;
  return 0;
}
