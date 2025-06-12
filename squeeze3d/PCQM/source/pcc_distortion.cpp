// Copyright (C) 2017, 2023, 2025 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <fstream>
#include <sstream>

#include "pcc_processing.hpp"
#include "pcc_distortion.hpp"

using namespace std;
using namespace pcc_quality;
using namespace nanoflann;

#define PRINT_TIMING 0

/*
   ***********************************************
   Implementation of local functions
   ***********************************************
 */

/**!
 * \function
 *   Compute the minimum and maximum NN distances, find out the
 *   intrinsic resolutions
 * \parameters
 *   @param cloudA: point cloud
 *   @param minDist: output
 *   @param maxDist: output
 * \note
 *   PointT typename of point used in point cloud
 * \author
 *   Dong Tian, MERL
 */
void
findNNdistances(PccPointCloud &cloudA, double &minDist, double &maxDist)
{
  typedef vector< vector<double> > my_vector_of_vectors_t;
  typedef KDTreeVectorOfVectorsAdaptor< my_vector_of_vectors_t, double >  my_kd_tree_t;

  maxDist =  numeric_limits<double>::min();
  minDist =  numeric_limits<double>::max();
  double distTmp = 0;
  mutex myMutex;

  my_kd_tree_t mat_index(3, cloudA.xyz.p, 10); // dim, cloud, max leaf

#pragma omp parallel for
  for (long i = 0; i < cloudA.size; ++i)
  {
    // cout << "*** " << i << endl;
    // do a knn search
    const size_t num_results = 3;
    vector<size_t> indices(num_results);
    vector<double> sqrDist(num_results);

    KNNResultSet<double> resultSet(num_results);

    resultSet.init( &indices[0], &sqrDist[0] );
    mat_index.index->findNeighbors( resultSet, &cloudA.xyz.p[i][0], SearchParams(10) );

    if (indices[0] != i || sqrDist[1] <= 0.0000000001)
    {
      // Print some warnings
      // cerr << "Error! nFound = " << nFound << ", i, iFound = " << i << ", " << indices[0] << ", " << indices[1] << endl;
      // cerr << "       Distances = " << sqrDist[0] << ", " << sqrDist[1] << endl;
      // cerr << "  Some points are repeated!" << endl;
    }

    else
    {
      // Use the second one. assume the first one is the current point
      myMutex.lock();
      distTmp = sqrt( sqrDist[1] );
      if (distTmp > maxDist)
        maxDist = distTmp;
      if (distTmp < minDist)
        minDist = distTmp;
      myMutex.unlock();
    }
  }
}

/**!
 * \function
 *   Convert the MSE error to PSNR numbers
 * \parameters
 *   @param cloudA:  the original point cloud
 *   @param dist2: the sqr of the distortion
 *   @param p: the peak value for conversion
 *   @param factor: default 1.0. For geometry errors value 3.0 should be provided
 * \return
 *   psnr value
 * \note
 *   PointT typename of point used in point cloud
 * \author
 *   Dong Tian, MERL
 */
float
getPSNR(float dist2, float p, float factor = 1.0)
{
  float max_energy = p * p;
  float psnr = 10 * log10( (factor * max_energy) / dist2 );

  return psnr;
}

/**!
 * \function
 *   Derive the normals for the decoded point cloud based on the
 *   normals in the original point cloud
 * \parameters
 *   @param cloudA:  the original point cloud
 *   @param cloudNormalsA: the normals in the original point cloud
 *   @param cloudB:  the decoded point cloud
 *   @param cloudNormalsB: the normals in the original point
 *     cloud. Output parameter
 * \note
 *   PointT typename of point used in point cloud
 * \author
 *   Dong Tian, MERL
 */
void
scaleNormals(PccPointCloud &cloudA, PccPointCloud &cloudNormalsA, PccPointCloud &cloudB, PccPointCloud &cloudNormalsB)
{
  // Prepare the buffer to compute the average normals
#if PRINT_TIMING
  clock_t t1 = clock();
#endif

  cloudNormalsB.normal.init(cloudB.size);
  vector< vector<int> > vecMap( cloudB.size );

  for (long i = 0; i < cloudB.size; i++)
  {
    vecMap[i].clear();
  }

  typedef vector< vector<double> > my_vector_of_vectors_t;
  typedef KDTreeVectorOfVectorsAdaptor< my_vector_of_vectors_t, double >  my_kd_tree_t;

  my_kd_tree_t mat_indexA(3, cloudA.xyz.p, 10); // dim, cloud, max leaf
  my_kd_tree_t mat_indexB(3, cloudB.xyz.p, 10); // dim, cloud, max leaf

  for (long i = 0; i < cloudA.size; i++)
  {

    const size_t num_results = 1;
    vector<size_t> indices(num_results);
    vector<double> sqrDist(num_results);

    KNNResultSet<double> resultSet(num_results);

    resultSet.init( &indices[0], &sqrDist[0] );
    mat_indexB.index->findNeighbors( resultSet, &cloudA.xyz.p[i][0], SearchParams(10) );

    cloudNormalsB.normal.n[indices[0]][0] += cloudNormalsA.normal.n[i][0];
    cloudNormalsB.normal.n[indices[0]][1] += cloudNormalsA.normal.n[i][1];
    cloudNormalsB.normal.n[indices[0]][2] += cloudNormalsA.normal.n[i][2];
    vecMap[ indices[0] ].push_back( i );
  }

  // average now
  for (long i = 0; i < cloudB.size; i++)
  {
    int nCount = vecMap[i].size();
    if (nCount > 0)      // main branch
    {
      cloudNormalsB.normal.n[i][0] = cloudNormalsB.normal.n[i][0] / nCount;
      cloudNormalsB.normal.n[i][1] = cloudNormalsB.normal.n[i][1] / nCount;
      cloudNormalsB.normal.n[i][2] = cloudNormalsB.normal.n[i][2] / nCount;
    }
    else
    {
      const size_t num_results = 1;
      vector<size_t> indices(num_results);
      vector<double> sqrDist(num_results);

      KNNResultSet<double> resultSet(num_results);

      resultSet.init( &indices[0], &sqrDist[0] );
      mat_indexA.index->findNeighbors( resultSet, &cloudB.xyz.p[i][0], SearchParams(10) );

      cloudNormalsB.normal.n[i][0] = cloudNormalsA.normal.n[indices[0]][0];
      cloudNormalsB.normal.n[i][1] = cloudNormalsA.normal.n[indices[0]][1];
      cloudNormalsB.normal.n[i][2] = cloudNormalsA.normal.n[indices[0]][2];
    }
  }

  // Set the flag
  cloudNormalsB.bNormal = true;

#if PRINT_TIMING
  clock_t t2 = clock();
  cout << "   Converting normal vector DONE. It takes " << (t2-t1)/CLOCKS_PER_SEC << " seconds (in CPU time)." << endl;
#endif
}

/**
   \brief helper function to convert RGB to YUV, using BT.601
*/
void
convertRGBtoYUV(const vector<unsigned char>  &in_rgb, float *out_yuv)
{
  // color space conversion to YUV
  out_yuv[0] = float( ( 0.299 * in_rgb[0] + 0.587 * in_rgb[1] + 0.114 * in_rgb[2]) / 255.0 );
  out_yuv[1] = float( (-0.147 * in_rgb[0] - 0.289 * in_rgb[1] + 0.436 * in_rgb[2]) / 255.0 );
  out_yuv[2] = float( ( 0.615 * in_rgb[0] - 0.515 * in_rgb[1] - 0.100 * in_rgb[2]) / 255.0 );
}

/**
   \brief helper function to convert RGB to YUV, using BT.709 formula
*/
void
convertRGBtoYUV_BT709(const vector<unsigned char>  &in_rgb, float *out_yuv)
{
  // color space conversion to YUV
  out_yuv[0] = float( ( 0.2126 * in_rgb[0] + 0.7152 * in_rgb[1] + 0.0722 * in_rgb[2]) / 255.0 );
  out_yuv[1] = float( (-0.1146 * in_rgb[0] - 0.3854 * in_rgb[1] + 0.5000 * in_rgb[2]) / 255.0 + 0.5000 );
  out_yuv[2] = float( ( 0.5000 * in_rgb[0] - 0.4542 * in_rgb[1] - 0.0458 * in_rgb[2]) / 255.0 + 0.5000 );
}

/**!
 * \function
 *   To compute "one-way" quality metric: Point-to-Point, Point-to-Plane
 *   and RGB. Loop over each point in A. Normals in B to be used
 *
 *   1) For each point in A, find a corresponding point in B.
 *   2) Form an error vector between the point pair.
 *   3) Use the length of the error vector as point-to-point measure
 *   4) Project the error vector along the normals in B, use the length
 *   of the projected error vector as point-to-plane measure
 *
 *   @param cloudA: Reference point cloud. e.g. the original cloud, on
 *     which normals would be estimated. It is the full set of point
 *     cloud. Multiple points in count
 *   @param cloudB: Processed point cloud. e.g. the decoded cloud
 *   @param cPar: Command line parameters
 *   @param cloudNormalsB: Normals for cloudB
 *   @param metric: updated quality metric, to be returned
 * \note
 *   PointT typename of point used in point cloud
 * \author
 *   Dong Tian, MERL
 */
void
findMetric(PccPointCloud &cloudA, PccPointCloud &cloudB, commandPar &cPar, PccPointCloud &cloudNormalsB, qMetric &metric)
{
  mutex myMutex;

#if PRINT_TIMING
  clock_t t2 = clock();
#endif
  double max_dist_b_c2c = std::numeric_limits<double>::min();
  double sse_dist_b_c2c = 0;
  double max_dist_b_c2p = std::numeric_limits<double>::min();
  double sse_dist_b_c2p = 0;
  double sse_reflectance = 0;
  long num = 0;

  double sse_color[3];
  sse_color[0] = sse_color[1] = sse_color[2] = 0.0;

  typedef vector< vector<double> > my_vector_of_vectors_t;
  typedef KDTreeVectorOfVectorsAdaptor< my_vector_of_vectors_t, double >  my_kd_tree_t;

  my_kd_tree_t mat_indexB(3, cloudB.xyz.p, 10); // dim, cloud, max leaf

#pragma omp parallel for
  for (long i = 0; i < cloudA.size; i++)
  {
    // Find the nearest neighbor in B. store it in 'j'
    const size_t num_results = 1;
    vector<size_t> indices(num_results);
    vector<double> sqrDist(num_results);

    KNNResultSet<double> resultSet(num_results);

    resultSet.init( &indices[0], &sqrDist[0] );
    mat_indexB.index->findNeighbors( resultSet, &cloudA.xyz.p[i][0], SearchParams(10) );

    int j = indices[0];
    if (j < 0)
      continue;

    // Compute the error vector
    vector<double> errVector(3);
    errVector[0] = cloudA.xyz.p[i][0] - cloudB.xyz.p[j][0];
    errVector[1] = cloudA.xyz.p[i][1] - cloudB.xyz.p[j][1];
    errVector[2] = cloudA.xyz.p[i][2] - cloudB.xyz.p[j][2];

    // Compute point-to-point, which should be equal to sqrt( sqrDist[0] )
    double distProj_c2c = errVector[0] * errVector[0] + errVector[1] * errVector[1] + errVector[2] * errVector[2];

    // Compute point-to-plane
    // Normals in B will be used for point-to-plane
    double distProj = 0.0;
    if (!cPar.c2c_only && cloudNormalsB.bNormal)
    {
      if ( !isnan( cloudNormalsB.normal.n[j][0] ) && !isnan( cloudNormalsB.normal.n[j][1] ) && !isnan( cloudNormalsB.normal.n[j][2] ) )
      {
        distProj = ( errVector[0] * cloudNormalsB.normal.n[j][0] +
                     errVector[1] * cloudNormalsB.normal.n[j][1] +
                     errVector[2] * cloudNormalsB.normal.n[j][2] );
        distProj *= distProj;  // power 2 for MSE
      }
      else
        distProj = errVector[0] * errVector[0] + errVector[1] * errVector[1] + errVector[2] * errVector[2];
    }

    double distColor[3];
    distColor[0] = distColor[1] = distColor[2] = 0.0;
    if (cPar.bColor && cloudA.bRgb && cloudB.bRgb)
    {
      float out[3];
      float in[3];

      convertRGBtoYUV_BT709(cloudA.rgb.c[i],in);
      convertRGBtoYUV_BT709(cloudB.rgb.c[j],out);

      distColor[0] = (in[0] - out[0]) * (in[0] - out[0]);
      distColor[1] = (in[1] - out[1]) * (in[1] - out[1]);
      distColor[2] = (in[2] - out[2]) * (in[2] - out[2]);
    }

    double distReflectance;
    distReflectance = 0.0;
    if (cPar.bLidar && cloudA.bLidar && cloudB.bLidar)
    {
      distReflectance = ( cloudA.lidar.reflectance[i] - cloudB.lidar.reflectance[j] ) * ( cloudA.lidar.reflectance[i] - cloudB.lidar.reflectance[j] );
    }

    myMutex.lock();

    num++;
    // mean square distance
    sse_dist_b_c2c += distProj_c2c;
    if (distProj_c2c > max_dist_b_c2c)
      max_dist_b_c2c = distProj_c2c;
    if (!cPar.c2c_only)
    {
      sse_dist_b_c2p += distProj;
      if (distProj > max_dist_b_c2p)
        max_dist_b_c2p = distProj;
    }
    if (cPar.bColor)
    {
      sse_color[0] += distColor[0];
      sse_color[1] += distColor[1];
      sse_color[2] += distColor[2];
    }
    if (cPar.bLidar && cloudA.bLidar && cloudB.bLidar)
    {
      sse_reflectance += distReflectance;
    }

    myMutex.unlock();
  }

  metric.c2p_mse = float( sse_dist_b_c2p / num );
  metric.c2c_mse = float( sse_dist_b_c2c / num );
  metric.c2p_hausdorff = float( max_dist_b_c2p );
  metric.c2c_hausdorff = float( max_dist_b_c2c );

  // from distance to PSNR. cloudA always the original
  metric.c2c_psnr = getPSNR( metric.c2c_mse, metric.pPSNR, 3 );
  metric.c2p_psnr = getPSNR( metric.c2p_mse, metric.pPSNR, 3 );
  metric.c2c_hausdorff_psnr = getPSNR( metric.c2c_hausdorff, metric.pPSNR, 3 );
  metric.c2p_hausdorff_psnr = getPSNR( metric.c2p_hausdorff, metric.pPSNR, 3 );

  if (cPar.bColor)
  {
    metric.color_mse[0] = float( sse_color[0] / num );
    metric.color_mse[1] = float( sse_color[1] / num );
    metric.color_mse[2] = float( sse_color[2] / num );

    metric.color_psnr[0] = getPSNR( metric.color_mse[0], 1.0 );
    metric.color_psnr[1] = getPSNR( metric.color_mse[1], 1.0 );
    metric.color_psnr[2] = getPSNR( metric.color_mse[2], 1.0 );
  }

  if (cPar.bLidar)
  {
    metric.reflectance_mse = float( sse_reflectance / num );
    metric.reflectance_psnr = getPSNR( float( metric.reflectance_mse ), float( std::numeric_limits<unsigned short>::max() ) );
  }

#if PRINT_TIMING
  clock_t t3 = clock();
  cout << "   Error computing takes " << (t3-t2)/CLOCKS_PER_SEC << " seconds (in CPU time)." << endl;
#endif
}

/*
   ***********************************************
   Implementation of exposed functions and classes
   ***********************************************
 */

/**!
 * **************************************
 *  Class commandPar
 *
 *  Dong Tian <tian@merl.com>
 * **************************************
 */

commandPar::commandPar()
{
  file1 = ""; file2 = "";
  normIn = "";
  singlePass = false;
  hausdorff = false;
  c2c_only = false;
  bColor = false;
  bLidar = false;

  resolution = 0.0;
}

/**!
 * **************************************
 *  Class qMetric
 *
 *  Dong Tian <tian@merl.com>
 * **************************************
 */

qMetric::qMetric()
{
  c2c_mse = 0; c2c_hausdorff = 0;
  c2p_mse = 0; c2p_hausdorff = 0;

  color_mse[0] = color_mse[1] = color_mse[2] = 0.0;
  color_psnr[0] = color_psnr[1] = color_psnr[2] = 0.0;
}

/**!
 * **************************************
 *  Function computeQualityMetric
 *
 *  Dong Tian <tian@merl.com>
 * **************************************
 */

/**!
 * function to compute the symmetric quality metric: Point-to-Point and Point-to-Plane
 *   @param cloudA: point cloud, original version
 *   @param cloudNormalA: point cloud normals, original version
 *   @param cloudB: point cloud, decoded/reconstructed version
 *   @param cPar: input parameters
 *   @param qual_metric: quality metric, to be returned
 *
 * \author
 *   Dong Tian, MERL
 */
void
pcc_quality::computeQualityMetric(PccPointCloud &cloudA, PccPointCloud &cloudNormalsA, PccPointCloud &cloudB, commandPar &cPar, qMetric &qual_metric)
{
  float pPSNR;

  if (cPar.resolution != 0.0)
  {
    cout << "Imported intrinsic resoluiton: " << cPar.resolution << endl;
    pPSNR = cPar.resolution;
  }
  else                          // Compute the peak value on the fly
  {
    double minDist;
    double maxDist;
    findNNdistances(cloudA, minDist, maxDist);
    pPSNR = float( maxDist );
    cout << "Minimum and maximum NN distances (intrinsic resolutions): " << minDist << ", " << maxDist << endl;
  }

  cout << "Peak distance for PSNR: " << pPSNR << endl;
  qual_metric.pPSNR = pPSNR;

  if (cPar.file2 != "")
  {
    // Check cloud size
    size_t orgSize = cloudA.size;
    size_t newSize = cloudB.size;
    float ratio = float(1.0) * newSize / orgSize;
    cout << "Point cloud sizes for org version, dec version, and the scaling ratio: " << orgSize << ", " << newSize << ", " << ratio << endl;
  }

  if (cPar.file2 == "" ) // If no file2 provided, return just after checking the NN
    return;

  // Based on normals on original point cloud, derive normals on reconstructed point cloud
  PccPointCloud cloudNormalsB;
  if (!cPar.c2c_only)
    scaleNormals( cloudA, cloudNormalsA, cloudB, cloudNormalsB );
  cout << "Normals prepared." << endl;
  cout << endl;

  // Use "a" as reference
  cout << "1. Use infile1 (A) as reference, loop over A, use normals on B. (A->B).\n";
  qMetric metricA;
  metricA.pPSNR = pPSNR;
  findMetric( cloudA, cloudB, cPar, cloudNormalsB, metricA );

  cout << "   mse1      (p2point): " << metricA.c2c_mse << endl;
  cout << "   mse1,PSNR (p2point): " << metricA.c2c_psnr << endl;
  if (!cPar.c2c_only)
  {
    cout << "   mse1      (p2plane): " << metricA.c2p_mse << endl;
    cout << "   mse1,PSNR (p2plane): " << metricA.c2p_psnr << endl;
  }
  if ( cPar.hausdorff )
  {
    cout << "   h.       1(p2point): " << metricA.c2c_hausdorff << endl;
    cout << "   h.,PSNR  1(p2point): " << metricA.c2c_hausdorff_psnr << endl;
    if (!cPar.c2c_only)
    {
      cout << "   h.       1(p2plane): " << metricA.c2p_hausdorff << endl;
      cout << "   h.,PSNR  1(p2plane): " << metricA.c2p_hausdorff_psnr << endl;
    }
  }
  if ( cPar.bColor )
  {
    cout << "   c[0],    1         : " << metricA.color_mse[0] << endl;
    cout << "   c[1],    1         : " << metricA.color_mse[1] << endl;
    cout << "   c[2],    1         : " << metricA.color_mse[2] << endl;
    cout << "   c[0],PSNR1         : " << metricA.color_psnr[0] << endl;
    cout << "   c[1],PSNR1         : " << metricA.color_psnr[1] << endl;
    cout << "   c[2],PSNR1         : " << metricA.color_psnr[2] << endl;
  }
  if ( cPar.bLidar )
  {
    cout << "   r,       1         : " << metricA.reflectance_mse  << endl;
    cout << "   r,PSNR   1         : " << metricA.reflectance_psnr << endl;
  }

  if (!cPar.singlePass)
  {
    // Use "b" as reference
    cout << "2. Use infile2 (B) as reference, loop over B, use normals on A. (B->A).\n";
    qMetric metricB;
    metricB.pPSNR = pPSNR;
    findMetric( cloudB, cloudA, cPar, cloudNormalsA, metricB );

    cout << "   mse2      (p2point): " << metricB.c2c_mse << endl;
    cout << "   mse2,PSNR (p2point): " << metricB.c2c_psnr << endl;
    if (!cPar.c2c_only)
    {
      cout << "   mse2      (p2plane): " << metricB.c2p_mse << endl;
      cout << "   mse2,PSNR (p2plane): " << metricB.c2p_psnr << endl;
    }
    if ( cPar.hausdorff )
    {
      cout << "   h.       2(p2point): " << metricB.c2c_hausdorff << endl;
      cout << "   h.,PSNR  2(p2point): " << metricB.c2c_hausdorff_psnr << endl;
      if (!cPar.c2c_only)
      {
        cout << "   h.       2(p2plane): " << metricB.c2p_hausdorff << endl;
        cout << "   h.,PSNR  2(p2plane): " << metricB.c2p_hausdorff_psnr << endl;
      }
    }
    if ( cPar.bColor)
    {
      cout << "   c[0],    2         : " << metricB.color_mse[0] << endl;
      cout << "   c[1],    2         : " << metricB.color_mse[1] << endl;
      cout << "   c[2],    2         : " << metricB.color_mse[2] << endl;
      cout << "   c[0],PSNR2         : " << metricB.color_psnr[0] << endl;
      cout << "   c[1],PSNR2         : " << metricB.color_psnr[1] << endl;
      cout << "   c[2],PSNR2         : " << metricB.color_psnr[2] << endl;
    }
    if ( cPar.bLidar )
    {
      cout << "   r,       2         : " << metricB.reflectance_mse  << endl;
      cout << "   r,PSNR   2         : " << metricB.reflectance_psnr << endl;
    }

    // Derive the final symmetric metric
    qual_metric.c2c_mse = max( metricA.c2c_mse, metricB.c2c_mse );
    qual_metric.c2p_mse = max( metricA.c2p_mse, metricB.c2p_mse );
    qual_metric.c2c_psnr = min( metricA.c2c_psnr, metricB.c2c_psnr );
    qual_metric.c2p_psnr = min( metricA.c2p_psnr, metricB.c2p_psnr );

    qual_metric.c2c_hausdorff = max( metricA.c2c_hausdorff, metricB.c2c_hausdorff	);
    qual_metric.c2p_hausdorff = max( metricA.c2p_hausdorff, metricB.c2p_hausdorff );
    qual_metric.c2c_hausdorff_psnr = min( metricA.c2c_hausdorff_psnr, metricB.c2c_hausdorff_psnr	);
    qual_metric.c2p_hausdorff_psnr = min( metricA.c2p_hausdorff_psnr, metricB.c2p_hausdorff_psnr );

    if ( cPar.bColor )
    {
      qual_metric.color_mse[0] = max( metricA.color_mse[0], metricB.color_mse[0] );
      qual_metric.color_mse[1] = max( metricA.color_mse[1], metricB.color_mse[1] );
      qual_metric.color_mse[2] = max( metricA.color_mse[2], metricB.color_mse[2] );

      qual_metric.color_psnr[0] = min( metricA.color_psnr[0], metricB.color_psnr[0] );
      qual_metric.color_psnr[1] = min( metricA.color_psnr[1], metricB.color_psnr[1] );
      qual_metric.color_psnr[2] = min( metricA.color_psnr[2], metricB.color_psnr[2] );
    }
    if ( cPar.bLidar )
    {
      qual_metric.reflectance_mse  = max( metricA.reflectance_mse,  metricB.reflectance_mse  );
      qual_metric.reflectance_psnr = min( metricA.reflectance_psnr, metricB.reflectance_psnr );
    }

    cout << "3. Final (symmetric).\n";
    cout << "   mseF      (p2point): " << qual_metric.c2c_mse << endl;
    cout << "   mseF,PSNR (p2point): " << qual_metric.c2c_psnr << endl;
    if (!cPar.c2c_only)
    {
      cout << "   mseF      (p2plane): " << qual_metric.c2p_mse << endl;
      cout << "   mseF,PSNR (p2plane): " << qual_metric.c2p_psnr << endl;
    }
    if ( cPar.hausdorff )
    {
      cout << "   h.        (p2point): " << qual_metric.c2c_hausdorff << endl;
      cout << "   h.,PSNR   (p2point): " << qual_metric.c2c_hausdorff_psnr << endl;
      if (!cPar.c2c_only)
      {
        cout << "   h.        (p2plane): " << qual_metric.c2p_hausdorff << endl;
        cout << "   h.,PSNR   (p2plane): " << qual_metric.c2p_hausdorff_psnr << endl;
      }
    }
    if ( cPar.bColor )
    {
      cout << "   c[0],    F         : " << qual_metric.color_mse[0] << endl;
      cout << "   c[1],    F         : " << qual_metric.color_mse[1] << endl;
      cout << "   c[2],    F         : " << qual_metric.color_mse[2] << endl;
      cout << "   c[0],PSNRF         : " << qual_metric.color_psnr[0] << endl;
      cout << "   c[1],PSNRF         : " << qual_metric.color_psnr[1] << endl;
      cout << "   c[2],PSNRF         : " << qual_metric.color_psnr[2] << endl;
    }
    if ( cPar.bLidar )
    {
      cout << "   r,       F         : " << qual_metric.reflectance_mse  << endl;
      cout << "   r,PSNR   F         : " << qual_metric.reflectance_psnr << endl;
    }
  }
}
