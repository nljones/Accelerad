# NLJ addition: OPTIX compiler flag
option(OptiX_USE "Include the option to use optix when compiling." ON)

qt5_wrap_cpp(rvu_moc rvuwidget.h mainwindow.h)
qt5_wrap_ui(rvu_ui mainwindow.ui)
qt5_wrap_ui(exposure_dialog_ui exposuredialog.ui)
qt5_wrap_ui(parameter_dialog_ui parameterdialog.ui)
qt5_wrap_ui(view_dialog_ui viewdialog.ui)
qt5_wrap_ui(increments_dialog_ui incrementsdialog.ui)
qt5_wrap_ui(commands_dialog_ui commandsdialog.ui)

include_directories(
  ${CMAKE_SOURCE_DIR}/src/rt
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

# NLJ addition
if(OptiX_USE AND WIN32)
  set(Qwt_DIR "C:/Qwt-6.1.3" CACHE PATH "Path to Qwt installed location.")
  include_directories("${Qwt_DIR}/include")
  link_directories("${Qwt_DIR}/lib")
  set(ACCELERAD_Qwt metricsplot.cxx)
  set(Qwt_LIBRARIES "${Qwt_DIR}/lib/qwt.lib")
  add_definitions("-DHAS_QWT")
endif()

add_library(qtrvu
  mainwindow.cxx
  qt.c
  qt_rvu_main.cxx
  rvuwidget.cxx
  ${ACCELERAD_Qwt}
  ${commands_dialog_ui}
  ${exposure_dialog_ui}
  ${increments_dialog_ui}
  ${parameter_dialog_ui}
  ${rvu_moc}
  ${rvu_ui}
  ${view_dialog_ui}
)
target_link_libraries(qtrvu ${Qwt_LIBRARIES} Qt5::Widgets)
