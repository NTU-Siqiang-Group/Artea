# Official Intel wheels for Linux x86-64: https://pypi.org/project/mkl-static/2025.3.0/
# Update the version, download paths and SHA256 checksums together.
# Paths are relative to https://files.pythonhosted.org/packages/; checksums cover each complete wheel.
set(mkl_version 2025.3.0)
set(mkl_packages mkl_static mkl_include mkl_devel onemkl_license)

# Static libraries.
set(mkl_static_download_path "a3/f3/9c18b8c144cfe9d99052c5aec77a19711f146cea97b2161a813ee9f96d7e")
set(mkl_static_sha256 "f05594dbe568b4fd6ff65fb7283631c78388d5ac4e6917cbf25dfd7dd37f6ed0")

# C/C++ headers.
set(mkl_include_download_path "d3/e7/5f8f78044b421f859f1a9f6aa4cc957c4733bee3e716a1d54b79797f241c")
set(mkl_include_sha256 "8fd38f51f543ce7031cf454a199529986277eeaf5b0e6a9ff8bbe27757b017e6")

# Intel's CMake and pkg-config files.
set(mkl_devel_download_path "9d/00/0e64ede5280b6fba32837e54e56a63219f3add42ed0ff101c9d2f6c5959f")
set(mkl_devel_sha256 "4704d62e1eae0e5d7281244ebd8c78aae4683490f51f8cceb512729c8e2c035d")

# Intel license and third-party notices.
set(onemkl_license_download_path "88/11/b43e8cde058c368ce7f8f9b1ca9f812f7397e4309148da7d24cb6b81b513")
set(onemkl_license_sha256 "a810b25bb24a90db4492d81c71cde13183d8951ce26960d6dc56d4f4e3dc95ff")
