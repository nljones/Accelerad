set(VERSION_FILE "${radiance_BINARY_DIR}/src/rt/Version.c")
create_version_file("${VERSION_FILE}")

# NLJ addition: OPTIX compiler flag
option(OptiX_USE "Include the option to use optix when compiling." ON)

add_library(radiance
  ambcomp.c ambient.c ambio.c aniso.c ashikhmin.c data.c dielectric.c fprism.c freeobjmem.c 
  func.c glass.c initotypes.c m_alias.c m_brdf.c m_bsdf.c m_clip.c m_direct.c
  m_mirror.c m_mist.c mx_func.c mx_data.c noise3.c normal.c o_face.c o_cone.c
  o_instance.c o_mesh.c p_data.c p_func.c preload.c raytrace.c renderopts.c
  source.c sphere.c srcobstr.c srcsupp.c srcsamp.c t_data.c t_func.c text.c
  "${VERSION_FILE}" virtuals.c)

if(WIN32)
  set(rayp_SOURCES raypwin.c)
else()
  set(rayp_SOURCES raypcalls.c)
endif()

add_library(raycalls raycalls.c ${rayp_SOURCES} rayfifo.c)

# NLJ addition
if(OptiX_USE)
  # This enforces a particular version of CMake that we require to process the script files
  # properly.
  cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)
  
  # As of CMake 2.6 policies were introduced in order to provide a mechanism for
  # adding backwards compatibility one feature at a time.  We will just specify
  # that all policies will use version 2.6 symantics.
  cmake_policy(VERSION 2.6)

  # Turn off the warning that NVCC issues when generating PTX from our CUDA samples.  This
  # is a custom extension to the FindCUDA code distributed by CMake.
  option(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)

  # Find at least a 2.3 version of CUDA.
  find_package(CUDA 2.3 REQUIRED)

  # Suggest a path to the OptiX SDK directory.
  # This line is taken from FindOptiX.cmake,
  # located in C:\ProgramData\NVIDIA Corporation\OptiX SDK 3.6.2\SDK\CMake.
  set(OptiX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 3.6.2" CACHE PATH "Path to OptiX installed location.")

  # Add OptiX CMake files to the module path
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${OptiX_INSTALL_DIR}/SDK/CMake)

  # Search for the OptiX libraries and include files.
  find_package(OptiX REQUIRED)

  # Add OptiX header files
  include_directories(${OptiX_INCLUDE} ${OptiX_INCLUDE}/optixu ${CUDA_INCLUDE_DIRS})

  # These calls will group PTX and CUDA files into their own directories in the Visual
  # Studio projects.
  source_group("PTX Files"  REGULAR_EXPRESSION ".+\\.ptx$")
  source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")

  # Add some useful default arguments to the nvcc flags.  This is an example of how we use
  # PASSED_FIRST_CONFIGURE.  Once you have configured, this variable is TRUE and following
  # block of code will not be executed leaving you free to edit the values as much as you
  # wish from the GUI or from ccmake.
  if(NOT PASSED_FIRST_CONFIGURE)
    set(flag "--use_fast_math")
    list(FIND CUDA_NVCC_FLAGS ${flag} index)
    if(index EQUAL -1)
      list(APPEND CUDA_NVCC_FLAGS ${flag})
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
    endif()
	
    set(flag "--gpu-architecture sm_20")
    list(FIND CUDA_NVCC_FLAGS ${flag} index)
    if(index EQUAL -1)
      list(APPEND CUDA_NVCC_FLAGS ${flag})
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
    endif()
	
    # Now that everything is done, indicate that we have finished configuring at least once.
    # We use this variable to set certain defaults only on the first pass, so that we don't
    # continually set them over and over again.
    set(PASSED_FIRST_CONFIGURE ON CACHE INTERNAL "Already Configured once?")
  endif(NOT PASSED_FIRST_CONFIGURE)

  # Include source wrappers for OptiX and CUDA.
  set(OptiX_SOURCES optix_radiance.c optix_ambient.c optix_util.c)
  
  # Compile the CUDA files to OBJ format.
  # This is a macro found in FindCUDA.cmake that compiles the listed
  # CUDA files into OBJ files with the prefix "cuda_compile_generated_"
  # in a subdirectory of the directory specified by CUDA_GENERATED_OUTPUT_DIR.
  CUDA_COMPILE(generated_CUDA_files cuda_adaptive_seeding.cu cuda_kmeans.cu)

  # Compile the OptiX CUDA files to PTX format.
  # This is a macro found in FindCUDA.cmake that compiles the listed
  # CUDA files into PTX files with the prefix "cuda_compile_ptx_generated_"
  # in the directory specified by CUDA_GENERATED_OUTPUT_DIR.
  CUDA_COMPILE_PTX(generated_PTX_files_rtrace sensor.cu)
  CUDA_COMPILE_PTX(generated_PTX_files_rpict camera.cu)
  CUDA_COMPILE_PTX(generated_PTX_files
    triangle_mesh.cu background.cu skybright.cu perezlum.cu
    radiance_normal.cu radiance_glass.cu radiance_light.cu
    ambient_records.cu ambient_generator.cu ambient_normal.cu
    ambient_cloud_generator.cu point_cloud_generator.cu point_cloud_normal.cu hemisphere_generator.cu)
endif()

add_executable(rtrace rtmain.c rtrace.c duphead.c persist.c ${OptiX_SOURCES} ${generated_CUDA_files} ${generated_PTX_files_rtrace} ${generated_PTX_files})
target_link_libraries(rtrace raycalls radiance rtrad ${optix_LIBRARY} ${optixu_LIBRARY} ${CUDA_CUDART_LIBRARY})

add_executable(rpict rpmain.c rpict.c srcdraw.c duphead.c persist.c ${OptiX_SOURCES} ${generated_CUDA_files} ${generated_PTX_files_rpict} ${generated_PTX_files})
target_link_libraries(rpict radiance rtrad ${optix_LIBRARY} ${optixu_LIBRARY} ${CUDA_CUDART_LIBRARY})

add_executable(lookamb lookamb.c ambio.c)
target_link_libraries(lookamb rtrad)

add_executable(rcontrib rcmain.c rcontrib.c rc2.c rc3.c)
target_link_libraries(rcontrib radiance rtrad)

set(targets_to_install lookamb radiance raycalls rtrace rpict rcontrib)

if(X11_FOUND)
  set(X11_SOURCES x11.c x11twind.c)
  add_definitions("-DHAS_X11")
endif()
if(QT4_FOUND)
  add_definitions("-DHAS_QT")
endif()
if (X11_FOUND OR QT4_FOUND)
  add_executable(rvu
    rvmain.c rview.c rv2.c rv3.c devtable.c devcomm.c editline.c ${X11_SOURCES}
    colortab.c)
  target_link_libraries(rvu raycalls radiance rtrad ${X11_LIBRARIES})
  list(APPEND targets_to_install rvu)

  # only link to rvu-qt-lib if qt is found
  if(QT4_FOUND)
    target_link_libraries(rvu qtrvu)
  endif()
endif()

add_test(rtrace_Version ${EXECUTABLE_OUTPUT_PATH}/rtrace -version)
set_tests_properties(rtrace_Version PROPERTIES
  PASS_REGULAR_EXPRESSION "RADIANCE")

install(TARGETS ${targets_to_install}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION bin
        ARCHIVE DESTINATION bin)
install(FILES rayinit.cal DESTINATION lib)

#install qt.conf & Qt plugins loaded by rvu at runtime
install(FILES ${QT_QGIF_${qt_plugin_term}_RELEASE} DESTINATION bin/plugins/imageformats)
install(FILES ${QT_QICO_${qt_plugin_term}_RELEASE} DESTINATION bin/plugins/imageformats)
install(FILES ${QT_QJPEG_${qt_plugin_term}_RELEASE} DESTINATION bin/plugins/imageformats)
install(FILES ${QT_QMNG_${qt_plugin_term}_RELEASE} DESTINATION bin/plugins/imageformats)
install(FILES ${QT_QSVG_${qt_plugin_term}_RELEASE} DESTINATION bin/plugins/imageformats)
install(FILES ${QT_QTIFF_${qt_plugin_term}_RELEASE} DESTINATION bin/plugins/imageformats)
install(FILES ${CMAKE_SOURCE_DIR}/InstallRules/qt.conf DESTINATION bin)

