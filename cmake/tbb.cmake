include_guard(GLOBAL)
include(FetchContent)

function(_artea_setup_tbb)
    set(BUILD_SHARED_LIBS ON)
    set(TBB_BUILD ON)
    set(TBBMALLOC_BUILD ON)
    set(TBBMALLOC_PROXY_BUILD OFF)
    set(TBB_TEST OFF)
    set(TBB_EXAMPLES OFF)
    set(TBB_STRICT OFF)
    set(TBB_INSTALL OFF)
    set(TBB_FIND_PACKAGE OFF)
    set(TBB_DISABLE_HWLOC_AUTOMATIC_SEARCH ON)

    # Use the benchmark's submodule; standalone Artea fetches the same pinned release.
    set(tbb_source_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../oneTBB")
    if(EXISTS "${tbb_source_dir}/CMakeLists.txt")
        add_subdirectory("${tbb_source_dir}" "${CMAKE_BINARY_DIR}/_deps/oneTBB-build")
    else()
        FetchContent_Declare(artea_tbb
            GIT_REPOSITORY https://github.com/uxlfoundation/oneTBB.git
            GIT_TAG f1862f38f83568d96e814e469ab61f88336cc595 # v2022.3.0
        )
        FetchContent_MakeAvailable(artea_tbb)
    endif()

    # TBB loads its allocator from the same library directory at runtime.
    add_dependencies(tbb tbbmalloc)
endfunction()

_artea_setup_tbb()
