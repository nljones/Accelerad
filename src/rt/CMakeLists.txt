set(VERSION_FILE "${radiance_BINARY_DIR}/src/rt/Version.c")
create_version_file("${VERSION_FILE}")

# NLJ addition: Simplify PTX file names
macro(PTX_SHROTEN ptx_short_names ptx_names)
  foreach(file ${ptx_names})
    get_filename_component(file_path "${file}" PATH)
    get_filename_component(file_name "${file}" NAME_WE)
	string(REGEX REPLACE "^cuda_compile_ptx_[0-9]*_?generated_" "" file_name "${file_name}")
	#string(REGEX REPLACE "\\.cu\\." "\\." file_name "${new_file}")
	set(new_file "${file_path}/${file_name}.ptx")
	add_custom_command(OUTPUT ${new_file} COMMAND ${CMAKE_COMMAND} -E copy "${file}" "${new_file}" DEPENDS "${file}")
	list(APPEND "${ptx_short_names}" "${new_file}")
  endforeach()
endmacro()

# NLJ addition: OPTIX compiler flag
option(OptiX_USE "Include the option to use optix when compiling." ON)

add_library(radiance
	ambcomp.c ambient.c ambio.c aniso.c ashikhmin.c data.c daysim.c dielectric.c fprism.c freeobjmem.c 
      	func.c glass.c initotypes.c m_alias.c m_brdf.c m_bsdf.c m_clip.c m_direct.c
      	m_mirror.c m_mist.c mx_func.c mx_data.c noise3.c normal.c o_face.c o_cone.c
      	o_instance.c o_mesh.c p_data.c p_func.c preload.c raytrace.c renderopts.c
      	source.c sphere.c srcobstr.c srcsupp.c srcsamp.c t_data.c t_func.c text.c
      	pmap.c pmapsrc.c pmapmat.c pmaprand.c pmapio.c 
      	pmapbias.c pmapparm.c pmapcontrib.c pmapamb.c pmapray.c pmapopt.c 
      	pmapdata.c pmapdiag.c pmaptype.c pmutil.c pmcontrib2.c
		../common/random.h ../common/platform.h func.c
      	"${VERSION_FILE}" virtuals.c ../common/paths.c)

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
  # located in C:\ProgramData\NVIDIA Corporation\OptiX SDK 3.9.2\SDK\CMake.
  if(WIN32)
    set(OptiX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 3.9.2" CACHE PATH "Path to OptiX installed location.")
  else()
    set(OptiX_INSTALL_DIR "~/NVIDIA-OptiX-SDK-3.9.2-linux64" CACHE PATH "Path to OptiX installed location.")
  endif()

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
	
    set(flag "-Xptxas -v")
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
  
  # This passes a preprocessor definition to cl.exe when processing CUDA code.
  if(USING_WINDOWS_CL)
    list(APPEND CUDA_NVCC_FLAGS --compiler-options /D_USE_MATH_DEFINES)
  endif()
  
  # Compile the CUDA files to OBJ format.
  # This is a macro found in FindCUDA.cmake that compiles the listed
  # CUDA files into OBJ files with the prefix "cuda_compile_generated_"
  # in a subdirectory of the directory specified by CUDA_GENERATED_OUTPUT_DIR.
  CUDA_COMPILE(generated_CUDA_files cuda_adaptive_seeding.cu cuda_kmeans.cu)

  # Compile the OptiX CUDA files to PTX format.
  # This is a macro found in FindCUDA.cmake that compiles the listed
  # CUDA files into PTX files with the prefix "cuda_compile_ptx_generated_"
  # in the directory specified by CUDA_GENERATED_OUTPUT_DIR.
  CUDA_COMPILE_PTX(generated_PTX_files_rtrace sensor.cu sensor_cloud_generator.cu irradiance_intersect.cu)
  CUDA_COMPILE_PTX(generated_PTX_files_rpict camera.cu point_cloud_generator.cu)
  CUDA_COMPILE_PTX(generated_PTX_files_rcontrib rcontrib_generator.cu tregenza.cu reinhart.cu klems_full.cu)
  CUDA_COMPILE_PTX(generated_PTX_files_rvu rvu_generator.cu diffuse_normal.cu)
  CUDA_COMPILE_PTX(generated_PTX_files
    triangle_mesh.cu background.cu skybright.cu perezlum.cu isotropsky.cu utah.cu gencumulativesky.cu
    radiance_normal.cu radiance_glass.cu radiance_light.cu source.cu fisheye.cu clip.cu
    ambient_records.cu ambient_normal.cu ambient_sample_generator.cu ambient_cloud_generator.cu
    point_cloud_normal.cu hemisphere_generator.cu)

  # Remove the prefix "cuda_compile_ptx_generated_" from PTX files.
  PTX_SHROTEN(PTX_files_rtrace "${generated_PTX_files_rtrace}")
  PTX_SHROTEN(PTX_files_rpict "${generated_PTX_files_rpict}")
  PTX_SHROTEN(PTX_files_rcontrib "${generated_PTX_files_rcontrib}")
  PTX_SHROTEN(PTX_files_rvu "${generated_PTX_files_rvu}")
  PTX_SHROTEN(PTX_files "${generated_PTX_files}")

  # Include source wrappers for OptiX and CUDA.
  # The empty blank.cpp file forces g++ to be used as the linker on Linux.
  set(ACCELERAD_LIBRARY accelerad)
  add_library(${ACCELERAD_LIBRARY} optix_radiance.c optix_ambient.c optix_util.c blank.cpp ${generated_CUDA_files} ${PTX_files})

  # Set source to be included in executables
  set(ACCELERAD_rtrace_files optix_rtrace.c ${PTX_files_rtrace})
  set(ACCELERAD_rpict_files optix_rpict.c ${PTX_files_rpict})
  set(ACCELERAD_rcontrib_files optix_rcontrib.c ${PTX_files_rcontrib})
  set(ACCELERAD_rvu_files optix_rvu.c ${PTX_files_rvu})
endif()

add_executable(rtrace rtmain.c rtrace.c duphead.c persist.c ${ACCELERAD_rtrace_files})
target_link_libraries(rtrace ${ACCELERAD_LIBRARY} ${optix_LIBRARY} ${CUDA_CUDART_LIBRARY} raycalls radiance rtrad)

add_executable(rpict rpmain.c rpict.c srcdraw.c duphead.c persist.c ${ACCELERAD_rpict_files})
target_link_libraries(rpict ${ACCELERAD_LIBRARY} ${optix_LIBRARY} ${CUDA_CUDART_LIBRARY} radiance rtrad)

add_executable(lookamb lookamb.c ambio.c)
target_link_libraries(lookamb rtrad)

add_executable(rcontrib rcmain.c rcontrib.c rc2.c rc3.c ${ACCELERAD_rcontrib_files})
target_link_libraries(rcontrib ${ACCELERAD_LIBRARY} ${optix_LIBRARY} ${CUDA_CUDART_LIBRARY} radiance rtrad)

add_executable(mkpmap mkpmap.c)
target_link_libraries(mkpmap radiance rtrad)
add_executable(pmapdump pmapdump.c pmaptype.c pmapparm.c)
target_link_libraries(pmapdump raycalls radiance rtrad)

set(targets_to_install lookamb radiance raycalls rtrace rpict rcontrib mkpmap pmapdump)

if(X11_FOUND)
  set(X11_SOURCES x11.c x11twind.c)
  add_definitions("-DHAS_X11")
endif()
if(Qt5Widgets_DIR)
  # set policy to link to qtmain
  cmake_policy(SET CMP0020 OLD)
  include_directories(${Qt5Widgets_INCLUDE_DIRS})
  add_definitions("-DHAS_QT")
endif()
if(NOT BUILD_HEADLESS)
  if (X11_FOUND OR Qt5Widgets_DIR)
    add_executable(rvu
      rvmain.c rview.c rv2.c rv3.c devtable.c devcomm.c editline.c ${X11_SOURCES}
      colortab.c ${ACCELERAD_rvu_files})
    target_link_libraries(rvu ${ACCELERAD_LIBRARY} ${optix_LIBRARY} ${CUDA_CUDART_LIBRARY} raycalls radiance rtrad ${X11_LIBRARIES})
    list(APPEND targets_to_install rvu)

    # only link to rvu-qt-lib if qt is found
    if(Qt5Widgets_DIR)
      target_link_libraries(rvu qtrvu)
    endif()
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

#install qt.conf & Qt plugins loaded by rvu at runtime. Ignore if we are building as headless
if(NOT BUILD_HEADLESS)
  get_target_property(GIF_PLUGIN Qt5::QGifPlugin LOCATION)
  get_target_property(ICO_PLUGIN Qt5::QICOPlugin LOCATION)
  get_target_property(JPEG_PLUGIN Qt5::QJpegPlugin LOCATION)
  #get_target_property(MNG_PLUGIN Qt5::QMngPlugin LOCATION)
  get_target_property(TIFF_PLUGIN Qt5::QTiffPlugin LOCATION)
  install(PROGRAMS "${GIF_PLUGIN}" DESTINATION bin/plugins/imageformats)
  install(PROGRAMS "${ICO_PLUGIN}" DESTINATION bin/plugins/imageformats)
  install(PROGRAMS "${JPEG_PLUGIN}" DESTINATION bin/plugins/imageformats)
  #install(PROGRAMS "${MNG_PLUGIN}" DESTINATION bin/plugins/imageformats)
  install(PROGRAMS "${TIFF_PLUGIN}" DESTINATION bin/plugins/imageformats)
  install(FILES ${CMAKE_SOURCE_DIR}/InstallRules/qt.conf DESTINATION bin)

  if(APPLE OR WIN32)
    if(WIN32)
      get_target_property(QPA_PLUGIN Qt5::QWindowsIntegrationPlugin LOCATION)
    elseif(APPLE)
      get_target_property(QPA_PLUGIN Qt5::QCocoaIntegrationPlugin LOCATION)
    endif()

    install(PROGRAMS "${QPA_PLUGIN}" DESTINATION bin/plugins/platforms)
  endif()
endif()
