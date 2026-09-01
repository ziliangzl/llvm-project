//===- SPMOps.cpp - SPM dialect ops ----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPM/SPMOps.h"
#include "SPM/SPMDialect.h"

using namespace mlir;
using namespace mlir::spm;

//===----------------------------------------------------------------------===//
// Shared load/store verification helpers.
//
// Mirrors `verifyMemoryOpTypes` from the upstream Triton `local_load` /
// `local_store` verifiers: the memory-domain descriptor and the
// compute-domain tensor must agree on shape and element type. No
// hardware-specific checks (alignment, addressability, ...) yet.
//===----------------------------------------------------------------------===//

template <typename MemDescType>
static LogicalResult verifyMemoryOpTypes(Operation *op, MemDescType memDescTy,
                                          RankedTensorType tensorType) {
  if (memDescTy.getElementType() != tensorType.getElementType())
    return op->emitOpError("buffer element type ")
           << memDescTy.getElementType()
           << " does not match tensor element type "
           << tensorType.getElementType();
  if (memDescTy.getShape() != tensorType.getShape())
    return op->emitOpError("buffer shape does not match tensor shape");
  return success();
}

//===----------------------------------------------------------------------===//
// L1LoadOp / L1StoreOp
//===----------------------------------------------------------------------===//

LogicalResult L1LoadOp::verify() {
  return verifyMemoryOpTypes(getOperation(), getSrc().getType(),
                              cast<RankedTensorType>(getResult().getType()));
}

LogicalResult L1StoreOp::verify() {
  if (!getDst().getType().getMutableMemory())
    return emitOpError("cannot store into an immutable buffer");
  return verifyMemoryOpTypes(getOperation(), getDst().getType(),
                              cast<RankedTensorType>(getSrc().getType()));
}

//===----------------------------------------------------------------------===//
// L2LoadOp / L2StoreOp
//===----------------------------------------------------------------------===//

LogicalResult L2LoadOp::verify() {
  return verifyMemoryOpTypes(getOperation(), getSrc().getType(),
                              cast<RankedTensorType>(getResult().getType()));
}

LogicalResult L2StoreOp::verify() {
  if (!getDst().getType().getMutableMemory())
    return emitOpError("cannot store into an immutable buffer");
  return verifyMemoryOpTypes(getOperation(), getDst().getType(),
                              cast<RankedTensorType>(getSrc().getType()));
}

#define GET_OP_CLASSES
#include "SPM/SPMOps.cpp.inc"
