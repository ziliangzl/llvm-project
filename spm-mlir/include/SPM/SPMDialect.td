//===- SPMDialect.td - SPM dialect ------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SPM_DIALECT
#define SPM_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// SPM dialect definition.
//===----------------------------------------------------------------------===//

def SPM_Dialect : Dialect {
  let name = "spm";
  let summary = "Experimentation playground for a from-scratch MLIR NPU backend.";
  let description = [{
    The `spm` dialect is a prototype for a new NPU backend built on MLIR.
    It currently models the NPU's two levels of software-managed on-chip
    SRAM (L1 and L2) as two separate descriptor types (`!spm.l1mem`,
    `!spm.l2mem`) with their own load/store ops. No hardware-specific
    layout, addressing, or bank-conflict constraints are modeled yet -
    those are expected to be layered on top once the hardware details are
    pinned down.
  }];
  let cppNamespace = "::mlir::spm";

  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

//===----------------------------------------------------------------------===//
// Base SPM operation definition.
//===----------------------------------------------------------------------===//

class SPM_Op<string mnemonic, list<Trait> traits = []> :
    Op<SPM_Dialect, mnemonic, traits>;

#endif // SPM_DIALECT
