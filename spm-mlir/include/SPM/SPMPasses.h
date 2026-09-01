//===- SPMPasses.h - SPM dialect passes ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SPM_SPMPASSES_H
#define SPM_SPMPASSES_H

#include "SPM/SPMDialect.h"
#include "SPM/SPMOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace spm {

/// Populates `patterns` with a pattern that lowers a `linalg.matmul`
/// operating directly on `!spm.l2mem` ins/outs into an explicit
/// `spm.l2_load` + `linalg.matmul` (on tensors) + `spm.l2_store` sequence.
void populateLowerLinalgMatmulToL2Patterns(RewritePatternSet &patterns);

#define GEN_PASS_DECL
#include "SPM/SPMPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "SPM/SPMPasses.h.inc"

} // namespace spm
} // namespace mlir

#endif // SPM_SPMPASSES_H
