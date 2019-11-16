#include <torch/extension.h>

void scatter_max(at::Tensor src, at::Tensor index, at::Tensor arg, int64_t dim) {
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scatter_max", &scatter_max, "Scatter Max (CPU)");
}
