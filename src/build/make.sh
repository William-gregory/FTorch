module unload cray-netcdf cray-hdf5 fre
module unload PrgEnv-pgi PrgEnv-intel PrgEnv-gnu PrgEnv-cray
module load PrgEnv-intel/8.5.0
module unload intel intel-classic intel-oneapi
module load intel-classic/2023.2.0
module load fre/bronx-21
module load cray-hdf5/1.12.2.11
module load libyaml/0.2.5

export KMP_STACKSIZE=512m
export NC_BLKSZ=1M
export F_UFMTENDIAN=big

export FI_VERBS_PREFER_XRC=0

# Platform environment overrides

       module unload cray-libsci
       module unload darshan-runtime
       module load git
       export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/lib64"


export PATH="/ncrc/home2/fms/local/opt/fre-commands/bronx-21/bin:${PATH}"

rm install_manifest.txt
rm libftorch.so
rm CMakeCache.txt
rm Makefile
rm cmake_install.cmake
rm -r CMakeFiles
rm -r modules
rm -r torch_v2p1_Inteloneapi2023p2p0

cmake  .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_Fortran_COMPILER=ftn \
  -DCMAKE_C_COMPILER=cc \
  -DCMAKE_CXX_COMPILER=CC \
  -DCMAKE_PREFIX_PATH=/ncrc/home2/William.Gregory/miniconda3/envs/ML/lib/python3.11/site-packages/torch \
  -DCMAKE_INSTALL_PREFIX=/gpfs/f5/gfdl_o/scratch/William.Gregory/FTorch/src/build/torch_v2p1_Inteloneapi2023p2p0 \
  -DENABLE_CUDA=FALSE

cmake --build .
cmake --install .

