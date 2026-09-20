include_guard(GLOBAL)
include(FetchContent)

set(ARTEA_MKL_ROOT "" CACHE PATH "Offline MKL prefix containing include/ and lib/cmake/mkl/")

function(_artea_setup_mkl)
    set(MKL_ROOT "${ARTEA_MKL_ROOT}")
    if(NOT MKL_ROOT)
        if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
            message(FATAL_ERROR "Automatic MKL download supports Linux x86-64; set ARTEA_MKL_ROOT otherwise.")
        endif()

        include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/mkl-packages.cmake)
        set(MKL_ROOT "${FETCHCONTENT_BASE_DIR}/mkl-${mkl_version}")
        foreach(package IN LISTS mkl_packages)
            set(wheel "${package}-${mkl_version}-py2.py3-none-manylinux_2_28_x86_64.whl")
            FetchContent_Declare(artea_${package}
                URL "https://files.pythonhosted.org/packages/${${package}_download_path}/${wheel}"
                DOWNLOAD_NAME "${package}-${mkl_version}.zip"
                URL_HASH "SHA256=${${package}_sha256}"
            )
            FetchContent_MakeAvailable(artea_${package})

            # Assemble one native prefix; hard links avoid duplicating the large static libraries.
            set(package_root "${artea_${package}_SOURCE_DIR}/${package}-${mkl_version}.data/data")
            file(GLOB_RECURSE package_files RELATIVE "${package_root}" "${package_root}/*")
            foreach(relative_path IN LISTS package_files)
                get_filename_component(parent_dir "${MKL_ROOT}/${relative_path}" DIRECTORY)
                file(MAKE_DIRECTORY "${parent_dir}")
                file(CREATE_LINK "${package_root}/${relative_path}"
                    "${MKL_ROOT}/${relative_path}" COPY_ON_ERROR)
            endforeach()
        endforeach()
        file(COPY_FILE
            "${artea_onemkl_license_SOURCE_DIR}/onemkl_license-${mkl_version}.dist-info/LICENSE.txt"
            "${MKL_ROOT}/share/doc/mkl/LICENSE.txt" ONLY_IF_DIFFERENT)
    endif()

    # RNG calls use independent streams on TBB workers; MKL needs no internal threading.
    # FORCE also replaces threading/linkage choices cached by older builds.
    set(MKL_THREADING sequential)
    set(MKL_LINK static)
    set(MKL_THREADING sequential CACHE INTERNAL "Artea uses sequential MKL" FORCE)
    set(MKL_LINK static CACHE INTERNAL "Artea links static MKL" FORCE)
    set(ENABLE_SYCL_COMPILER OFF)
    set(MKL_SYCL_THREADING sequential)
    set(MKL_INCLUDE "${MKL_ROOT}/include")
    set(MKL_DIR "${MKL_ROOT}/lib/cmake/mkl")
    find_package(MKL CONFIG REQUIRED PATHS "${MKL_DIR}" NO_DEFAULT_PATH)
endfunction()

_artea_setup_mkl()
