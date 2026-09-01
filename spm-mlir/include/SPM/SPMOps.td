//===- SPMOps.td - SPM dialect ops -------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SPM_OPS
#define SPM_OPS

include "SPM/SPMTypes.td"

//===----------------------------------------------------------------------===//
// L1 load/store.
//===----------------------------------------------------------------------===//

def SPM_L1LoadOp : SPM_Op<"l1_load"> {
  let summary = "Load a buffer out of L1 SRAM into the compute domain.";
  let description = [{
    Reads the data described by an `!spm.l1mem` value out of L1
    software-managed SRAM and produces it as a tensor value, i.e. moves it
    from the "memory domain" to the "compute domain".

    Example:

    ```mlir
    %0 = spm.l1_load %buf : !spm.l1mem<64x64xf32> -> tensor<64x64xf32>
    ```
  }];

  let arguments = (ins SPM_L1MemType:$src);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
  let hasVerifier = 1;
}

def SPM_L1StoreOp : SPM_Op<"l1_store"> {
  let summary = "Store a compute-domain value into L1 SRAM.";
  let description = [{
    Writes a tensor value into the buffer described by an `!spm.l1mem`
    value, i.e. moves it from the "compute domain" to the "memory domain".
    The destination must not be `immutable`.

    Example:

    ```mlir
    spm.l1_store %val, %buf : tensor<64x64xf32> -> !spm.l1mem<64x64xf32>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$src, SPM_L1MemType:$dst);
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `->` type($dst)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// L2 load/store.
//===----------------------------------------------------------------------===//

def SPM_L2LoadOp : SPM_Op<"l2_load"> {
  let summary = "Load a buffer out of L2 SRAM into the compute domain.";
  let description = [{
    See `spm.l1_load`; identical semantics, operating on `!spm.l2mem`.

    Example:

    ```mlir
    %0 = spm.l2_load %buf : !spm.l2mem<64x64xf32> -> tensor<64x64xf32>
    ```
  }];

  let arguments = (ins SPM_L2MemType:$src);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
  let hasVerifier = 1;
}

def SPM_L2StoreOp : SPM_Op<"l2_store"> {
  let summary = "Store a compute-domain value into L2 SRAM.";
  let description = [{
    See `spm.l1_store`; identical semantics, operating on `!spm.l2mem`.

    Example:

    ```mlir
    spm.l2_store %val, %buf : tensor<64x64xf32> -> !spm.l2mem<64x64xf32>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$src, SPM_L2MemType:$dst);
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `->` type($dst)";
  let hasVerifier = 1;
}

#endif // SPM_OPS
