cmake_minimum_required(VERSION 3.16)

project(image-texture-analysis)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(Qt6_DIR "/usr/local/opt/qt/lib/cmake/Qt6")

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

if (Eigen3_FOUND)
    INCLUDE_DIRECTORIES("${EIGEN3_INCLUDE_DIR}")
    message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
endif ()

find_package(OpenCV REQUIRED)

find_package(Qt6 COMPONENTS Widgets REQUIRED)
find_package(Qt6 COMPONENTS Concurrent REQUIRED)
find_package(Qt6 COMPONENTS Core REQUIRED)
find_package(Qt6 COMPONENTS Qml REQUIRED)
find_package(Qt6 COMPONENTS Quick REQUIRED)
find_package(Qt6 COMPONENTS Charts REQUIRED)
find_package(Qt6 COMPONENTS LinguistTools REQUIRED)

set(QT6_LIBS
        ${QT6_LIBS}
        Qt6::Charts
        Qt6::Quick
        Qt6::Qml
        Qt6::Widgets
        Qt6::Concurrent
        Qt6::Core)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(image-texture-analysis
        analysis/TextureAnalysis.cpp
        controller/PolygonController.cpp
        controller/RectController.cpp
        viewer/Viewer.cpp
        mainwindow.cpp
        main.cpp)

target_link_libraries(image-texture-analysis ${OpenCV_LIBS} ${QT6_LIBS})
