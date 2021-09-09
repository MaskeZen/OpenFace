# Local libraries
include_directories(
    ${LandmarkDetector_SOURCE_DIR}/include
    ./include
)

add_executable(FacePreDetectionImgDaemon FacePreDetectionImgDaemon.cpp daemon.cpp)
target_link_libraries(FacePreDetectionImgDaemon LandmarkDetector)
target_link_libraries(FacePreDetectionImgDaemon FaceAnalyser)
target_link_libraries(FacePreDetectionImgDaemon GazeAnalyser)
target_link_libraries(FacePreDetectionImgDaemon Utilities)

install (TARGETS FacePreDetectionImgDaemon DESTINATION bin)