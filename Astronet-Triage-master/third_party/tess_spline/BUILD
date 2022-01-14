package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

exports_files(["LICENSE"])

py_library(
    name = "tess_spline",
    srcs = ["tess_spline.py"],
    srcs_version = "PY2AND3",
    deps = ["//third_party/robust_mean"],
)

py_test(
    name = "tess_spline_test",
    size = "small",
    srcs = ["tess_spline_test.py"],
    srcs_version = "PY2AND3",
    deps = [":tess_spline"],
)
