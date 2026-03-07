# bin/

This directory contains symbolic links to compiled executables from `build/apps/`.

The symlinks are automatically created by CMake during the build process, allowing you to run applications directly from the project root:

```bash
./bin/probe_radius
./bin/build_conv_graph
```

instead of:

```bash
./build/apps/probe_radius
./build/apps/build_conv_graph
```
