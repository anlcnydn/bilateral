cmake_minimum_required(VERSION 2.8)
project( assignment2 )
find_package( /usr/local/opt/opencv3/share/OpenCV/OpenCV REQUIRED )
add_executable( bilateral_filter bilateral_filter.cpp)
target_link_libraries( bilateral_filter ${OpenCV_LIBS} )

