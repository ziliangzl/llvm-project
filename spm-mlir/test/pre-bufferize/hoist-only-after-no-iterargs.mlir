// RUN: mlir-opt %s -one-shot-bufferize | FileCheck %s

// Variant of hoist-only-after.mlir: the 5 slots are hoisted out of the loop
// nest but NOT threaded through iter_args (what a plain LICM-style hoist would
// produce). Measured result: One-Shot Bufferize still gives 5 allocs at
// function scope and 0 allocs / 0 extra copies inside the loop, because an
// alloc_tensor has undefined contents and every slot here is fully overwritten
// before it is read in the same iteration.
//
// See SPM_HOIST_ONLY_EXAMPLE.md for why the iter_args form is still the one we
// emit (accumulators, post-loop readers, and the slotCount = 2 rotation path
// all need the loop-carried version).

// CHECK-LABEL: func.func @fused_matmul_relu
// CHECK-COUNT-5: memref.alloc() alignment = 64 : memref<{{.*}}, 1>
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK-NOT:         memref.alloc
// CHECK:         return

#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @fused_matmul_relu(%arg0: tensor<128x256xf32>, %arg1: tensor<256x64xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128x64xf32>) -> tensor<128x64xf32> {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    // 5 hoisted L1 slots, NOT threaded through iter_args (naive hoist / LICM-style)
    %bias_l1 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<32xf32>
    %a_l1 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<32x256xf32>
    %b_l1 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<256x32xf32>
    %acc_l1 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<32x32xf32>
    %out_l1 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<32x32xf32>
    %r = scf.for %i = %c0 to %c128 step %c32
        iter_args(%out = %arg3) -> (tensor<128x64xf32>) {
      %rr = scf.for %j = %c0 to %c64 step %c32
          iter_args(%out1 = %out) -> (tensor<128x64xf32>) {
        %cs = tensor.extract_slice %arg2[%i] [32] [1] : tensor<128xf32> to tensor<32xf32>
        %bias_v = bufferization.materialize_in_destination %cs in %bias_l1 : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
        %as = tensor.extract_slice %arg0[%i, 0] [32, 256] [1, 1] : tensor<128x256xf32> to tensor<32x256xf32>
        %a_v = bufferization.materialize_in_destination %as in %a_l1 : (tensor<32x256xf32>, tensor<32x256xf32>) -> tensor<32x256xf32>
        %bs = tensor.extract_slice %arg1[0, %j] [256, 32] [1, 1] : tensor<256x64xf32> to tensor<256x32xf32>
        %b_v = bufferization.materialize_in_destination %bs in %b_l1 : (tensor<256x32xf32>, tensor<256x32xf32>) -> tensor<256x32xf32>
        %f = linalg.fill ins(%cst : f32) outs(%acc_l1 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %mm = linalg.matmul ins(%a_v, %b_v : tensor<32x256xf32>, tensor<256x32xf32>) outs(%f : tensor<32x32xf32>) -> tensor<32x32xf32>
        %g = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%bias_v, %mm : tensor<32xf32>, tensor<32x32xf32>) outs(%out_l1 : tensor<32x32xf32>) {
        ^bb0(%in: f32, %in_2: f32, %o: f32):
          %x = arith.addf %in_2, %in : f32
          %y = arith.maximumf %x, %cst : f32
          linalg.yield %y : f32
        } -> tensor<32x32xf32>
        %ins = tensor.insert_slice %g into %out1[%i, %j] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<128x64xf32>
        scf.yield %ins : tensor<128x64xf32>
      }
      scf.yield %rr : tensor<128x64xf32>
    }
    return %r : tensor<128x64xf32>
  }
}
