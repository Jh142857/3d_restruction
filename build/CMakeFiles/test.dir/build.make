# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cjh/Course/计算机视觉/3d_restruction

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cjh/Course/计算机视觉/3d_restruction/build

# Include any dependencies generated for this target.
include CMakeFiles/test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test.dir/flags.make

CMakeFiles/test.dir/src/test.cpp.o: CMakeFiles/test.dir/flags.make
CMakeFiles/test.dir/src/test.cpp.o: ../src/test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cjh/Course/计算机视觉/3d_restruction/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test.dir/src/test.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test.dir/src/test.cpp.o -c /home/cjh/Course/计算机视觉/3d_restruction/src/test.cpp

CMakeFiles/test.dir/src/test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test.dir/src/test.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cjh/Course/计算机视觉/3d_restruction/src/test.cpp > CMakeFiles/test.dir/src/test.cpp.i

CMakeFiles/test.dir/src/test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test.dir/src/test.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cjh/Course/计算机视觉/3d_restruction/src/test.cpp -o CMakeFiles/test.dir/src/test.cpp.s

# Object files for target test
test_OBJECTS = \
"CMakeFiles/test.dir/src/test.cpp.o"

# External object files for target test
test_EXTERNAL_OBJECTS =

../out/test: CMakeFiles/test.dir/src/test.cpp.o
../out/test: CMakeFiles/test.dir/build.make
../out/test: /usr/local/lib/libopencv_dnn.so.4.5.0
../out/test: /usr/local/lib/libopencv_gapi.so.4.5.0
../out/test: /usr/local/lib/libopencv_highgui.so.4.5.0
../out/test: /usr/local/lib/libopencv_ml.so.4.5.0
../out/test: /usr/local/lib/libopencv_objdetect.so.4.5.0
../out/test: /usr/local/lib/libopencv_photo.so.4.5.0
../out/test: /usr/local/lib/libopencv_stitching.so.4.5.0
../out/test: /usr/local/lib/libopencv_video.so.4.5.0
../out/test: /usr/local/lib/libopencv_videoio.so.4.5.0
../out/test: /usr/local/lib/libopencv_imgcodecs.so.4.5.0
../out/test: /usr/local/lib/libopencv_calib3d.so.4.5.0
../out/test: /usr/local/lib/libopencv_features2d.so.4.5.0
../out/test: /usr/local/lib/libopencv_flann.so.4.5.0
../out/test: /usr/local/lib/libopencv_imgproc.so.4.5.0
../out/test: /usr/local/lib/libopencv_core.so.4.5.0
../out/test: CMakeFiles/test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cjh/Course/计算机视觉/3d_restruction/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../out/test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test.dir/build: ../out/test

.PHONY : CMakeFiles/test.dir/build

CMakeFiles/test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test.dir/clean

CMakeFiles/test.dir/depend:
	cd /home/cjh/Course/计算机视觉/3d_restruction/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cjh/Course/计算机视觉/3d_restruction /home/cjh/Course/计算机视觉/3d_restruction /home/cjh/Course/计算机视觉/3d_restruction/build /home/cjh/Course/计算机视觉/3d_restruction/build /home/cjh/Course/计算机视觉/3d_restruction/build/CMakeFiles/test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test.dir/depend

