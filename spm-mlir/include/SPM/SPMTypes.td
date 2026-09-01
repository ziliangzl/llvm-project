//===- SPMTypes.td - SPM dialect types --------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SPM_TYPES
#define SPM_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "SPM/SPMDialect.td"

class SPM_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<SPM_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Shared-memory descriptor base class.
//
// Both SRAM levels are modeled as independent types (see
// NPU_LOAD_STORE_MEMDESC_PORTING.md discussion): they happen to carry the
// same fields today, but are deliberately kept as separate C++/IR types
// rather than one type parameterized by a memory-space attribute, so that
// each level's verifier/lowering can diverge freely later without the two
// levels ever being (accidentally or intentionally) interchangeable in the
// IR.
//===----------------------------------------------------------------------===//

class SPM_MemDescTypeBase<string name, string typeMnemonic>
    : SPM_Type<name, typeMnemonic, [ShapedTypeInterface]> {
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "bool":$mutableMemory,
    ArrayRefParameter<"int64_t">:$allocShape
  );

  // No `encoding`/layout field yet: v0 assumes a single implicit
  // row-major layout for every buffer. Hardware-specific addressing
  // (bank conflicts, swizzling, alignment, ...) is intentionally deferred
  // until the underlying SRAM constraints are known.
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  // ShapedTypeInterface: shapes here are always static, so this is always
  // ranked. `getShape`/`getElementType` are already provided by the
  // `shape`/`elementType` parameters above; `cloneWith` is implemented
  // out-of-line per concrete type in SPMTypes.cpp.
  let extraClassDeclaration = [{
    bool hasRank() const { return true; }
    ::mlir::ShapedType cloneWith(::std::optional<::llvm::ArrayRef<int64_t>> shape,
                                  ::mlir::Type elementType) const;
  }];
}

def SPM_L1MemType : SPM_MemDescTypeBase<"L1Mem", "l1mem"> {
  let summary = "Descriptor for a buffer living in L1 software-managed SRAM.";
  let description = [{
    `!spm.l1mem` describes a buffer allocated in L1, the smaller/closer
    level of software-managed on-chip SRAM. It carries:

    - `shape`: the logical shape this view of the buffer exposes.
    - `elementType`: the element type of the buffer.
    - `mutableMemory`: whether the buffer may be written to.
    - `allocShape`: the shape of the underlying allocation this descriptor
      was carved out of. Equal to `shape` for a plain buffer; wider than
      `shape` for e.g. a slice of a multi-buffered allocation. Kept
      separate from `shape` from day one so that multi-buffering /
      software-pipelining schemes have somewhere to record the full
      allocation later without a type-level redesign.

    Example:

    ```mlir
    !spm.l1mem<64x64xf32>
    !spm.l1mem<64x64xf32, immutable>
    !spm.l1mem<64x64xf32, alloc<3x64x64>>
    ```
  }];
}

def SPM_L2MemType : SPM_MemDescTypeBase<"L2Mem", "l2mem"> {
  let summary = "Descriptor for a buffer living in L2 software-managed SRAM.";
  let description = [{
    Same shape as `!spm.l1mem` (see its description for the field
    semantics), but for the L2 level of software-managed on-chip SRAM.

    Example:

    ```mlir
    !spm.l2mem<64x64xf32>
    ```
  }];
}

#endif // SPM_TYPES
