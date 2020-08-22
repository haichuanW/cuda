function(cc_library)
  cmake_parse_arguments(CC_LIB
    ""
    "NAME"
    "HDRS;SRCS;COPTS;DEFINES;LINKOPTS;DEPS"
    ${ARGN}
  )

  set(_NAME "${CC_LIB_NAME}")

  # Check if this is a header-only library
  # Note that as of February 2019, many popular OS's (for example, Ubuntu
  # 16.04 LTS) only come with cmake 3.5 by default.  For this reason, we can't
  # use list(FILTER...)
  set(CC_SRCS "${CC_LIB_SRCS}")
  foreach(src_file IN LISTS CC_SRCS)
    if(${src_file} MATCHES ".*\\.(h|inc)")
      list(REMOVE_ITEM CC_SRCS "${src_file}")
    endif()
  endforeach()
  if("${CC_SRCS}" STREQUAL "")
    set(CC_LIB_IS_INTERFACE 1)
  else()
    set(CC_LIB_IS_INTERFACE 0)
  endif()

  if(NOT CC_LIB_IS_INTERFACE)
    add_library(${_NAME} STATIC "")
    target_sources(${_NAME} PRIVATE ${CC_LIB_SRCS} ${CC_LIB_HDRS})
    target_include_directories(${_NAME}
      PUBLIC
        "$<BUILD_INTERFACE:${U3V_COMMON_INCLUDE_DIRS}>"
    )
    target_compile_options(${_NAME}
      PRIVATE ${CC_LIB_COPTS})
    target_link_libraries(${_NAME}
      PUBLIC ${CC_LIB_DEPS}
      PRIVATE
        ${CC_LIB_LINKOPTS}
    )
    target_compile_definitions(${_NAME} PUBLIC ${CC_LIB_DEFINES})
    set_target_properties(${_NAME} PROPERTIES
      CXX_STANDARD ${U3V_CXX_STANDARD}
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
    )
  else()
    add_library(${_NAME} INTERFACE)
    target_include_directories(${_NAME}
      INTERFACE
        "$<BUILD_INTERFACE:${U3V_COMMON_INCLUDE_DIRS}>"
    )
    target_link_libraries(${_NAME}
      INTERFACE
        ${CC_LIB_DEPS}
        ${CC_LIB_LINKOPTS}
    )
    target_compile_definitions(${_NAME} INTERFACE ${CC_LIB_DEFINES})
  endif()
endfunction()

function(cc_binary)
  cmake_parse_arguments(CC_LIB
    ""
    "NAME"
    "HDRS;SRCS;COPTS;DEFINES;LINKOPTS;DEPS"
    ${ARGN}
  )

  set(_NAME "${CC_LIB_NAME}")

  add_executable(${_NAME} ${CC_LIB_SRCS} ${CC_LIB_HDRS})
  target_include_directories(${_NAME}
    PUBLIC
      "$<BUILD_INTERFACE:${U3V_COMMON_INCLUDE_DIRS}>"
  )
  target_compile_options(${_NAME}
    PRIVATE ${CC_LIB_COPTS})
  target_link_libraries(${_NAME}
    PUBLIC ${CC_LIB_DEPS}
    PRIVATE
      ${CC_LIB_LINKOPTS}
  )
  target_compile_definitions(${_NAME} PUBLIC ${CC_LIB_DEFINES})
  set_target_properties(${_NAME} PROPERTIES
    CXX_STANDARD ${U3V_CXX_STANDARD}
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )
endfunction()
