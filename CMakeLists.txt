cmake_minimum_required(VERSION 3.6)

project(3dsmc_project)

# Options
option(TESTS_ENABLED "Whether to build test binaries" OFF)

set(CMAKE_CXX_STANDARD 14)
set(LIBRARY_DIR C:/Users/PaddlePad/Documents/UNIVERSITAET/3DScan/Aufgaben/Libs CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")
set(glog_DIR ${LIBRARY_DIR}/Glog/lib/cmake/glog CACHE PATH "Path to installed glog")
set(Ceres_DIR ${LIBRARY_DIR}/Ceres/lib/cmake/Ceres CACHE PATH "Path to installed Ceres")
set(OpenCV_DIR C:/Users/PaddlePad/Documents/UNIVERSITAET/AR/Resources/opencv/build/x64/vc15/lib CACHE PATH "Path to installed OpenCV")

find_package(Eigen3 REQUIRED)
find_package(glog REQUIRED)
find_package(Ceres REQUIRED)
#ADDED
find_package(OpenCV REQUIRED)
find_package(colmap REQUIRED)

add_subdirectory(external/json)
include_directories(external/spdlog/include)

get_target_property(GLOG_DLL_PATH_DEBUG glog::glog IMPORTED_LOCATION_DEBUG)
get_target_property(GLOG_DLL_PATH_RELEASE glog::glog IMPORTED_LOCATION_RELEASE)

add_definitions("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")

set(SOURCE_FILES 
    src/AppConfig.cpp
    src/BundleAdjuster.cpp
    src/main.cpp
)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/linclude/eigen3)

add_executable(3dsmc_project ${SOURCE_FILES})
target_link_libraries(3dsmc_project colmap::colmap ceres Eigen3::Eigen ${OpenCV_LIBS} nlohmann_json::nlohmann_json)
target_include_directories(3dsmc_project PUBLIC ${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

include_directories(${COLMAP_INCLUDE_DIRS})
add_executable(bundle_adjustment src/bundle_adjustment.cpp src/BundleAdjuster.cpp)
target_link_libraries(bundle_adjustment colmap::colmap ceres Eigen3::Eigen ${OpenCV_LIBS} nlohmann_json::nlohmann_json)

add_library(bundle_adjuster src/BundleAdjuster.cpp )
target_link_libraries(bundle_adjuster ceres Eigen3::Eigen)

if (TESTS_ENABLED)
    include(FetchContent)
    FetchContent_Declare(
    googletest
    # Specify the commit you depend on and update it regularly.
    URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # Now simply link against gtest or gtest_main as needed. Eg
    add_compile_definitions(GLOG_USE_GLOG_EXPORT)
    add_executable(test test/bundle_adjuster_tests.cpp )
    target_link_libraries(test gtest_main  bundle_adjuster)
    target_include_directories(test PUBLIC ${EIGEN3_INCLUDE_DIR})
    add_test(NAME 3dsmc_test COMMAND test)
    target_include_directories(test PUBLIC ${OpenCV_INCLUDE_DIRS})

endif()

# On Windows copy dll to executable folder
if(WIN32)
    # Visual Studio properties
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT 3dsmc_project)
    set_property(TARGET 3dsmc_project PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

    # On Windows copy dll to executable folder
    #add_custom_command(TARGET 3dsmc_project POST_BUILD
        #COMMAND ${CMAKE_COMMAND} -E copy_if_different
            #"${FreeImage_LIBRARY_DIR}/FreeImage.dll"
            #${CMAKE_BINARY_DIR})
    
    add_custom_command(TARGET 3dsmc_project POST_BUILD        
        COMMAND ${CMAKE_COMMAND} -E copy_if_different          
            "${GLOG_DLL_PATH_DEBUG}"                           
            ${CMAKE_BINARY_DIR})                     
    
    add_custom_command(TARGET 3dsmc_project POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GLOG_DLL_PATH_RELEASE}"
            ${CMAKE_BINARY_DIR})
endif(WIN32)

