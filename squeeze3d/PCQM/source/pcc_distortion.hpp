// Copyright (C) 2017, 2023, 2025 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PCC_DISTORTION_HPP
#define PCC_DISTORTION_HPP

#include <iostream>
#include <fstream>
#include <mutex>

#include "nanoflann/nanoflann.hpp"

using namespace nanoflann;
#include "nanoflann/KDTreeVectorOfVectorsAdaptor.h"

#include "pcc_processing.hpp"

using namespace std;
using namespace pcc_processing;

#define PCC_QUALITY_VERSION "0.10"

namespace pcc_quality {

  /**!
   * \brief
   *  Command line parameters
   *
   *  Dong Tian <tian@merl.com>
   */
  class commandPar
  {
  public:
    string file1;
    string file2;

    string normIn;           //! input file name for normals, if it is different from the input file 1

    bool   singlePass; //! Force to run a single pass algorithm. where the loop is over the original point cloud

    bool   hausdorff;         //! true: output hausdorff metric as well

    bool   c2c_only;          //! skip point-to-plane metric
    bool   bColor;            //! check color distortion as well
    bool   bLidar;            //! report reflectance as well

    float  resolution;        //! intrinsic resolution, imported. for geometric distortion

    commandPar();
  };

  /**!
   * \brief
   *  Store the quality metric for point to plane measurements
   */
  class qMetric {

  public:
    // point-2-point ( cloud 2 cloud ), benchmark metric
    float c2c_mse;            //! store symm mse metric
    float c2c_hausdorff;      //! store symm haussdorf
    float c2c_psnr;
    float c2c_hausdorff_psnr; //! store symm haussdorf

    float color_mse[3];       //! color components, root mean square
    float color_psnr[3];      //! psnr

    // point-2-plane ( cloud 2 plane ), proposed metric
    float c2p_mse;            //! store symm mse metric
    float c2p_hausdorff;      //! store symm haussdorf
    float c2p_psnr;
    float c2p_hausdorff_psnr; //! store symm haussdorf

    // point 2 plane ( cloud 2 plane ), proposed metric
    float pPSNR; //! Peak value for PSNR computation. Intrinsic resolution

    // reflectance
    float reflectance_mse;
    float reflectance_psnr;

    qMetric();
  };

  /**!
   * \brief
   *
   *  Interface function to compute quality metrics
   *
   *  Dong Tian <tian@merl.com>
   */
  void computeQualityMetric( PccPointCloud &cloudA, PccPointCloud &cloudNormalsA, PccPointCloud &cloudB, commandPar &cPar, qMetric &qual_metric );

};

#endif
