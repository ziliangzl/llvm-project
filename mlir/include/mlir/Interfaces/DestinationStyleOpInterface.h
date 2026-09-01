//===- DestinationStyleOpInterface.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_DESTINATIONSTYLEOPINTERFACE_H_
#define MLIR_INTERFACES_DESTINATIONSTYLEOPINTERFACE_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace detail {
/// Verify that `op` conforms to the invariants of DestinationStyleOpInterface
LogicalResult verifyDestinationStyleOpInterface(Operation *op);

/// Returns "true" if `type` should be treated as a buffer (an in-memory
/// value that DPS inits write into in place) for destination-passing-style
/// purposes. This covers the builtin memref types as well as any other
/// non-tensor, non-vector type that opts in by implementing
/// `ShapedTypeInterface` -- e.g. a dialect-defined "descriptor" type for a
/// custom, non-builtin memory space that plays the same role a memref does
/// for its ops.
inline bool isDestinationStyleBufferLikeType(Type type) {
  // Tensors and vectors are SSA value types, never "memory", even though
  // they also implement ShapedTypeInterface.
  if (isa<TensorType, VectorType>(type))
    return false;
  return isa<ShapedType>(type);
}
} // namespace detail
} // namespace mlir

/// Include the generated interface declarations.
#include "mlir/Interfaces/DestinationStyleOpInterface.h.inc"

#endif // MLIR_INTERFACES_DESTINATIONSTYLEOPINTERFACE_H_
