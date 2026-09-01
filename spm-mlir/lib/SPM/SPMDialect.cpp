//===- SPMDialect.cpp - SPM dialect ----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPM/SPMDialect.h"
#include "SPM/SPMOps.h"
#include "SPM/SPMTypes.h"

using namespace mlir;
using namespace mlir::spm;

#include "SPM/SPMOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// SPM dialect.
//===----------------------------------------------------------------------===//

void SPMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "SPM/SPMOps.cpp.inc"
      >();
  registerTypes();
}
