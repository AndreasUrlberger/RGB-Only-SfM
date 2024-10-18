FROM linuxserver/code-server:latest

RUN apt-get -y update && apt-get install -y

# Install required dependencies
# <-
RUN apt-get -y install cmake libgflags-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev build-essential g++ gdb python3-dev autotools-dev libicu-dev libbz2-dev libboost-all-dev libflann-dev libfreeimage3 libfreeimage-dev lz4
# ->

# Configure GCC 9
# <-
RUN apt-get -y install gcc-9 g++-9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9
# ->

WORKDIR /
RUN mkdir build_image
WORKDIR /build_image

# Install Glog from source (Ceres)
# <-
RUN cd /tmp && \
    git clone https://github.com/google/glog.git && \
    cd glog && \
    cmake -S . -B build -G "Unix Makefiles" && \
    cmake --build build --target install && \
    rm -rf /tmp/glog
# ->

# Install Ceres Solver
# <-
RUN cd /tmp && \
    git clone https://ceres-solver.googlesource.com/ceres-solver && \
    cd ceres-solver && \ 
    mkdir build && \
    cd build && \
    cmake -DBUILD_TESTING=OFF .. && \
    make -j4 install && \
    rm -rf /tmp/ceres-solver
# ->

WORKDIR /config/workspace/

RUN apt-get install -y vim clang-format clangd

# opencv
RUN apt-get install -y wget unzip

# enable graphics 
RUN apt-get install -y libgtk2.0-dev libgtk-3-dev pkg-config 

# install
RUN cd /tmp && \ 
    wget https://github.com/opencv/opencv/archive/4.x.zip -O opencv.zip && \
    unzip opencv.zip && \
    mv opencv-4.x opencv && \
    cd opencv && \ 
    mkdir -p build && cd build && \
    cmake -DBUILD_TESTS=OFF  -DBUILD_PERF_TESTS=OFF -DBUILD_opencv_apps=OFF .. && \
    make -j4 && \
    make -j4 install && \
    rm -rf /tmp/opencv

RUN apt-get install -y ninja-build \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libceres-dev \
    libcgal-dev 

RUN cd /tmp && \
    git clone https://github.com/colmap/colmap.git && \
    cd colmap && \
    mkdir build && \
    cd build && \
    cmake -DGUI_ENABLED=0 -DOPENGL_ENABLED=0 .. && \
    make -j4 && \
    make install

# Build with: docker build . -t 3dsmc
