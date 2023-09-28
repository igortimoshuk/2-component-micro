cmake_minimum_required(VERSION 3.15)

###  Add optional RELEASE/DEBUG compile to flags
if(NOT TARGET flags)
    add_library(flags INTERFACE)
endif()
target_compile_options(flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-fno-strict-aliasing -fdiagnostics-color=always -Wall -Wpedantic -Wextra -Wconversion -Wunused>) 
target_compile_options(flags INTERFACE $<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CXX>>: ${MARCH} ${MTUNE}>)
target_compile_options(flags INTERFACE $<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CXX>>: -fno-omit-frame-pointer -fstack-protector -D_FORTIFY_SOURCE=2>)
target_compile_options(flags INTERFACE $<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:Clang>>: -fstandalone-debug>)
target_compile_options(flags INTERFACE $<$<AND:$<CONFIG:RELWITHDEBINFO>,$<COMPILE_LANGUAGE:CXX>>: -fno-omit-frame-pointer -fstack-protector -D_FORTIFY_SOURCE=2>)




