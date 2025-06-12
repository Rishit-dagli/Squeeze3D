<!--
Copyright (C) 2017, 2023, 2025 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: BSD-3-Clause
-->

# Point Cloud Quality Metric

## Introduction

It is challenging to measure the geometry distortion of point cloud introduced by point cloud compression. Conventionally, the errors between point clouds are measured in terms of point-to-point or point-to-surface distances, that either ignores the surface structures or heavily tends to rely on specific surface reconstructions. To overcome these drawbacks, we propose using point-to-plane distances as a measure of geometric distortions on point cloud compression. The intrinsic resolution of the point clouds is proposed as a normalizer to convert the mean square errors to PSNR numbers. In addition, the perceived local planes are investigated at different scales of the point cloud. Finally, the proposed metric is independent of the size of the point cloud and rather reveals the geometric fidelity of the point cloud. From experiments, we demonstrate that our method could better track the perceived quality than the point-to-point approach while requires limited computations.

## Installation

1. Dependency
   Boost (`BSL-1.0` license) is required to compile and run the program. In particular,
   component `program_options` in Boost is required to process command line
   parameters.

2. Compiling instructions
   `CMakeLists.txt` is provided for cmake to generate makefiles. General
   practice using cmake should be followed in order to compile the
   program. Suggested steps to compile under Debug mode are shown below.

```
   :$ cd /path/to/root/folder/of/the/source/code/

   :$ mkdir debug

   :$ cd debug

   :$ cmake -DCMAKE_BUILD_TYPE=Debug ../source
   # You may want to specify the location of a pariticular Boost version,
   # e.g., via -DBOOST_ROOT=/path/to/boost_1_63_0

   :$ make
   "pc_error_d" to be generated under ./test folder
```

3. Reference
   MPEG input document M40522, "Updates and Integration of Evaluation Metric Software for PCC"

## Citation

If you use the software, please cite the following  ([TR2017-113](https://merl.com/publications/TR2017-113)):

```
@inproceedings{Tian2017sep,
author = {Tian, Dong and Ochimizu, Hideaki and Feng, Chen and Cohen, Robert A. and Vetro, Anthony},
title = {Geometric Distortion Metrics for Point Cloud Compression},
booktitle = {IEEE International Conference on Image Processing (ICIP)},
year = 2017,
month = sep,
doi = {10.1109/ICIP.2017.8296925},
url = {https://www.merl.com/publications/TR2017-113}
}
```

## Contact

Anthony Vetro <avetro@merl.com>

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## License

Released under `BSD-3-Clause` license, as found in the [LICENSE.md](LICENSE.md) file.

All files, except as noted below:

```
Copyright (C) 2017, 2023, 2025 Mitsubishi Electric Research Laboratories (MERL).

SPDX-License-Identifier: BSD-3-Clause
```

`source/nanoflann/KDTreeVectorOfVectorsAdaptor.h` (see [BSD-2-Clause.txt](BSD-2-Clause.txt)):

```
Copyright 2011-16 Jose Luis Blanco (joseluisblancoc@gmail.com). All rights reserved.

SPDX-License-Identifier: BSD-2-Clause
```

`source/nanoflann/nanoflann.hpp` (see [BSD-2-Clause.txt](BSD-2-Clause.txt)):

```
Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
Copyright 2011-2016  Jose Luis Blanco (joseluisblancoc@gmail.com). All rights reserved.

SPDX-License-Identifier: BSD-2-Clause
```
