package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "tess_io",
    srcs = ["tess_io.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "median_filter",
    srcs = ["median_filter.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "median_filter_test",
    size = "small",
    srcs = ["median_filter_test.py"],
    srcs_version = "PY2AND3",
    deps = [":median_filter"],
)

py_library(
    name = "periodic_event",
    srcs = ["periodic_event.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "periodic_event_test",
    size = "small",
    srcs = ["periodic_event_test.py"],
    srcs_version = "PY2AND3",
    deps = [":periodic_event"],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":periodic_event",
        ":util",
    ],
)
