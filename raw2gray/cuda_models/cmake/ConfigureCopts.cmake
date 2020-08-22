list(APPEND U3V_GCC_FLAGS
    "-Wall"
    "-Wextra"
    "-Wcast-qual"
    "-Wconversion-null"
    "-Wmissing-declarations"
    "-Woverlength-strings"
    "-Wpointer-arith"
    "-Wunused-local-typedefs"
    "-Wunused-result"
    "-Wvarargs"
    "-Wvla"
    "-Wwrite-strings"
)

set(U3V_DEFAULT_COPTS "${U3V_GCC_FLAGS}")
set(U3V_CXX_STANDARD 14)
