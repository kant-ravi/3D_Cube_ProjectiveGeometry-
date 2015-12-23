cmake_minimum_required(VERSION 2.8)
project( MyProjects )
find_package( OpenCV REQUIRED )

add_executable( Cube3D Cube3D.cpp )
target_link_libraries( Cube3D ${OpenCV_LIBS} )

