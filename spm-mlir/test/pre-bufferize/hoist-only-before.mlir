// RUN: mlir-opt %s -one-shot-bufferize | FileCheck %s
// RUN: mlir-opt %s -one-shot-bufferize -buffer-loop-hoisting | FileCheck %s --check-prefix=HOISTED

// "Before" IR of the HoistOnly example (SPM_HOIST_ONLY_EXAMPLE.md), i.e. the
// output of hoist-only-input.mlir. 5 L1 alloc_tensor ops live inside the inner
// loop body -> after bufferization they become 5 memref.alloc INSIDE the loop
// nest, which design 5.8/Q1 and 7.2/V10 forbid.

// CHECK-LABEL: func.func @fused_matmul_relu
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK-COUNT-5:     memref.alloc() alignment = 64 : memref<{{.*}}, 1>
// CHECK:             scf.yield

// The memref-level upstream pass fixes the symptom after the fact: it hoists
// all 5 allocs to the function entry block (see the design doc for why we still
// do it at tensor level instead).
// HOISTED-LABEL: func.func @fused_matmul_relu
// HOISTED-COUNT-5: memref.alloc() alignment = 64 : memref<{{.*}}, 1>
// HOISTED:         scf.for
// HOISTED:           scf.for
// HOISTED-NOT:         memref.alloc
// HOISTED:         return

#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @fused_matmul_relu(%arg0: tensor<128x256xf32>, %arg1: tensor<256x64xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128x64xf32>) -> tensor<128x64xf32> {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = scf.for %arg4 = %c0 to %c128 step %c32 iter_args(%arg5 = %arg3) -> (tensor<128x64xf32>) {
      %1 = scf.for %arg6 = %c0 to %c64 step %c32 iter_args(%arg7 = %arg5) -> (tensor<128x64xf32>) {
        %extracted_slice = tensor.extract_slice %arg2[%arg4] [32] [1] : tensor<128xf32> to tensor<32xf32>
        %2 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<32xf32>
        %3 = bufferization.materialize_in_destination %extracted_slice in %2 : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
        %extracted_slice_0 = tensor.extract_slice %arg0[%arg4, 0] [32, 256] [1, 1] : tensor<128x256xf32> to tensor<32x256xf32>
        %4 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<32x256xf32>
        %5 = bufferization.materialize_in_destination %extracted_slice_0 in %4 : (tensor<32x256xf32>, tensor<32x256xf32>) -> tensor<32x256xf32>
        %extracted_slice_1 = tensor.extract_slice %arg1[0, %arg6] [256, 32] [1, 1] : tensor<256x64xf32> to tensor<256x32xf32>
        %6 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<256x32xf32>
        %7 = bufferization.materialize_in_destination %extracted_slice_1 in %6 : (tensor<256x32xf32>, tensor<256x32xf32>) -> tensor<256x32xf32>
        %8 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<32x32xf32>
        %9 = linalg.fill {__producer__} ins(%cst : f32) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %10 = linalg.matmul {__producer__} ins(%5, %7 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%9 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %11 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<32x32xf32>
        %12 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3, %10 : tensor<32xf32>, tensor<32x32xf32>) outs(%11 : tensor<32x32xf32>) attrs =  {__root__} {
        ^bb0(%in: f32, %in_2: f32, %out: f32):
          %13 = arith.addf %in_2, %in : f32
          %14 = arith.maximumf %13, %cst : f32
          linalg.yield %14 : f32
        } -> tensor<32x32xf32>
        %inserted_slice = tensor.insert_slice %12 into %arg7[%arg4, %arg6] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<128x64xf32>
        scf.yield %inserted_slice : tensor<128x64xf32>
      }
      scf.yield %1 : tensor<128x64xf32>
    }
    return %0 : tensor<128x64xf32>
  }
}
