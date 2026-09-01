//===- SPMTypes.cpp - SPM dialect types ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPM/SPMTypes.h"

#include "SPM/SPMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::spm;

//===----------------------------------------------------------------------===//
// Shared parse/print/verify helpers for the L1/L2 descriptor types.
//
// `!spm.<mnemonic><shape x elemType[, immutable][, alloc<allocShape>]>`,
// e.g. `!spm.l1mem<64x64xf32>` or `!spm.l1mem<64x64xf32, alloc<3x64x64>>`.
//===----------------------------------------------------------------------===//

template <typename ConcreteType>
static Type parseMemDescType(AsmParser &parser) {
  if (parser.parseLess())
    return {};

  SmallVector<int64_t> shape;
  Type elementType;
  if (parser.parseDimensionList(shape, /*allowDynamic=*/false) ||
      parser.parseType(elementType))
    return {};

  bool mutableMemory = true;
  SmallVector<int64_t> allocShape(shape.begin(), shape.end());
  bool sawAlloc = false;

  while (succeeded(parser.parseOptionalComma())) {
    if (succeeded(parser.parseOptionalKeyword("immutable"))) {
      mutableMemory = false;
      continue;
    }
    if (succeeded(parser.parseOptionalKeyword("alloc"))) {
      if (sawAlloc)
        return parser.emitError(parser.getCurrentLocation(),
                                 "duplicate 'alloc' specifier"),
               Type();
      sawAlloc = true;
      allocShape.clear();
      if (parser.parseLess() ||
          parser.parseDimensionList(allocShape, /*allowDynamic=*/false,
                                     /*withTrailingX=*/false) ||
          parser.parseGreater())
        return {};
      continue;
    }
    return parser.emitError(parser.getCurrentLocation(),
                             "expected 'immutable' or 'alloc'"),
           Type();
  }

  if (parser.parseGreater())
    return {};

  auto loc = parser.getCurrentLocation();
  return ConcreteType::getChecked(
      [&] { return parser.emitError(loc); }, parser.getContext(), shape,
      elementType, mutableMemory, allocShape);
}

template <typename ConcreteType>
static void printMemDescType(ConcreteType type, AsmPrinter &printer) {
  printer << "<";
  for (int64_t dim : type.getShape())
    printer << dim << "x";
  printer << type.getElementType();
  if (!type.getMutableMemory())
    printer << ", immutable";
  if (ArrayRef<int64_t> allocShape = type.getAllocShape();
      allocShape != type.getShape()) {
    printer << ", alloc<";
    llvm::interleave(allocShape, printer, "x");
    printer << ">";
  }
  printer << ">";
}

static LogicalResult
verifyMemDescType(function_ref<InFlightDiagnostic()> emitError,
                   ArrayRef<int64_t> shape, Type elementType,
                   bool mutableMemory, ArrayRef<int64_t> allocShape) {
  if (shape.empty())
    return emitError() << "expected a buffer of rank >= 1";
  for (int64_t dim : shape)
    if (dim <= 0)
      return emitError() << "buffer dimensions must be positive";
  if (!elementType.isIntOrFloat())
    return emitError() << "expected an integer or float element type";
  if (allocShape.size() < shape.size())
    return emitError() << "alloc shape must have at least as many "
                           "dimensions as the buffer shape";
  return success();
}

// ShapedTypeInterface::cloneWith. If a new shape is given, there is no way
// to know how it maps onto an existing multi-buffered `allocShape`, so the
// clone starts a fresh allocation equal to the new shape.
template <typename ConcreteType>
static ShapedType cloneMemDescType(ConcreteType type,
                                    std::optional<ArrayRef<int64_t>> shape,
                                    Type elementType) {
  if (!shape)
    return ConcreteType::get(type.getContext(), type.getShape(), elementType,
                              type.getMutableMemory(), type.getAllocShape());
  return ConcreteType::get(type.getContext(), *shape, elementType,
                            type.getMutableMemory(), *shape);
}

//===----------------------------------------------------------------------===//
// L1MemType
//===----------------------------------------------------------------------===//

Type L1MemType::parse(AsmParser &parser) {
  return parseMemDescType<L1MemType>(parser);
}

void L1MemType::print(AsmPrinter &printer) const {
  printMemDescType(*this, printer);
}

LogicalResult
L1MemType::verify(function_ref<InFlightDiagnostic()> emitError,
                    ArrayRef<int64_t> shape, Type elementType,
                    bool mutableMemory, ArrayRef<int64_t> allocShape) {
  return verifyMemDescType(emitError, shape, elementType, mutableMemory,
                            allocShape);
}

ShapedType L1MemType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                                  Type elementType) const {
  return cloneMemDescType(*this, shape, elementType);
}

//===----------------------------------------------------------------------===//
// L2MemType
//===----------------------------------------------------------------------===//

Type L2MemType::parse(AsmParser &parser) {
  return parseMemDescType<L2MemType>(parser);
}

void L2MemType::print(AsmPrinter &printer) const {
  printMemDescType(*this, printer);
}

LogicalResult
L2MemType::verify(function_ref<InFlightDiagnostic()> emitError,
                    ArrayRef<int64_t> shape, Type elementType,
                    bool mutableMemory, ArrayRef<int64_t> allocShape) {
  return verifyMemDescType(emitError, shape, elementType, mutableMemory,
                            allocShape);
}

ShapedType L2MemType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                                  Type elementType) const {
  return cloneMemDescType(*this, shape, elementType);
}

//===----------------------------------------------------------------------===//
// TableGen'd type method definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "SPM/SPMOpsTypes.cpp.inc"

void SPMDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "SPM/SPMOpsTypes.cpp.inc"
      >();
}
