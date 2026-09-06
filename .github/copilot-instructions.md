# tensor-kernel-engine

- Keep Python paths CPU-safe and make CUDA/Triton acceleration optional.
- Preserve public APIs in `src/ops` when optimizing kernels.
- Run `ruff check .` and `pytest` after Python changes.
- Run `cmake -S . -B build && cmake --build build` when C++ changes are made.
