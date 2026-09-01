# -*- Python -*-

import os

import lit.formats

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

config.name = "SPM"

config.test_format = lit.formats.ShTest()

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.spm_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

config.excludes = ["Inputs", "Examples", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

config.spm_tools_dir = os.path.join(config.spm_obj_root, "bin")
config.spm_libs_dir = os.path.join(config.spm_obj_root, "lib")

config.substitutions.append(("%spm_libs", config.spm_libs_dir))

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.spm_tools_dir, config.llvm_tools_dir]
tools = [
    "spm-opt",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
