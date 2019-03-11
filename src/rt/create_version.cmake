#${CMAKE_COMMAND} -DRADIANCE_VERSION=v -DVERSION_OUT_FILE=v -DVERSION_IN_FILE=src/rt/VERSION -DVERSION_GOLD=src/rt/Version.c -P src/common/create_version.cmake

# if the gold version exists then use that instead
if(EXISTS "${VERSION_GOLD}")
  configure_file("${VERSION_GOLD}" "${VERSION_OUT_FILE}" COPYONLY)
  return()
endif()

find_program(DATE date)
if(DATE)
  execute_process(COMMAND ${DATE} "+%F"
    OUTPUT_VARIABLE DATE_STR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
elseif(WIN32)
  execute_process(COMMAND "cmd" " /C date /T"
    OUTPUT_VARIABLE DATE_STR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  #string(REGEX REPLACE "(..)/(..)/..(..).*" "\\1/\\2/\\3" ${RESULT} ${${RESULT}})
else()
  message(SEND_ERROR "date not implemented")
  set(DATE_STR 000000)
endif()
find_program(WHO whoami)
if(WHO)
  execute_process(COMMAND ${WHO}
    OUTPUT_VARIABLE WHO_STR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()
find_program(HOSTNAME hostname)
if(HOSTNAME)
  execute_process(COMMAND ${HOSTNAME}
    OUTPUT_VARIABLE HOST_STR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()

file(READ "${VERSION_IN_FILE}" VERSION)
string(STRIP "${VERSION}" VERSION)
set(CONTENTS "Accelerad ${VERSION} lastmod ${DATE_STR} by ${WHO_STR} on ${HOST_STR} (based on RADIANCE ${RADIANCE_VERSION} by G. Ward)")
message("${CONTENTS}")
string(REPLACE "\\" "\\\\" CONTENTS "${CONTENTS}") # look for instances of the escape character
file(WRITE "${VERSION_OUT_FILE}" "char VersionID[]=\"${CONTENTS}\";\nchar VersionShortID[]=\"${VERSION}\";\n")
