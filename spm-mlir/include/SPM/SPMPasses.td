//===- SPMPasses.td - SPM dialect passes -------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SPM_PASSES
#define SPM_PASSES

include "mlir/Pass/PassBase.td"

def SPMLowerLinalgMatmulToL2 : Pass<"spm-lower-linalg-matmul-to-l2", "::mlir::func::FuncOp"> {
  let summary = "Lower linalg.matmul on !spm.l2mem operands to l2_load/linalg.matmul/l2_store";
  let description = [{
    Matches a `linalg.matmul` whose `ins`/`outs` operands are all
    `!spm.l2mem` (i.e. it operates directly on L2 SRAM buffers, the same
    way a memref-based `linalg.matmul` operates directly on shared/global
    memory) and rewrites it into an explicit load/compute/store sequence:

    ```mlir
    linalg.matmul ins(%a, %b : !spm.l2mem<...>, !spm.l2mem<...>)
                  outs(%c : !spm.l2mem<...>)
    ```

    becomes

    ```mlir
    %a_t = spm.l2_load %a : !spm.l2mem<...> -> tensor<...>
    %b_t = spm.l2_load %b : !spm.l2mem<...> -> tensor<...>
    %c_t = spm.l2_load %c : !spm.l2mem<...> -> tensor<...>
    %r = linalg.matmul ins(%a_t, %b_t : tensor<...>, tensor<...>)
                        outs(%c_t : tensor<...>) -> tensor<...>
    spm.l2_store %r, %c : tensor<...> -> !spm.l2mem<...>
    ```
  }];
  let dependentDialects = [
    "::mlir::spm::SPMDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

#endif // SPM_PASSES
