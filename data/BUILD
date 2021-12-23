package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "generate_input_records",
    srcs = ["generate_input_records.py"],
    deps = [":preprocess"],
)

py_binary(
    name = "preprocess",
    srcs = ["preprocess.py"],
    deps = [
        "//light_curve_util:tess_io",
        "//light_curve_util:median_filter",
        "//light_curve_util:util",
        "//third_party/kepler_spline",
    ],
)

py_binary(
    name="make_catalog",
    srcs=["make_catalog.py"],
)

py_binary(
    name="make_empty_catalog",
    srcs=["make_empty_catalog.py"],
)
