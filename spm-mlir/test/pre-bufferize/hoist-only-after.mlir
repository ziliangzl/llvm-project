// RUN: mlir-opt %s -one-shot-bufferize | FileCheck %s

// "After" IR of the HoistOnly example (SPM_HOIST_ONLY_EXAMPLE.md):
// the 5 L1 slots are allocated once before the outermost loop and threaded
// through the iter_args of every enclosing loop (design 5.7, slotCount = 1).
//
// Expected bufferization result: 5 static L1 allocs at function scope,
// ZERO memref.alloc and ZERO extra memref.copy inside the loop nest
// (the 4 copies left are the 3 DRAM->L1 loads and the 1 L1->DRAM writeback).

// CHECK-LABEL: func.func @fused_matmul_relu
// CHECK:         memref.alloc() alignment = 64 : memref<32xf32, 1>
// CHECK:         memref.alloc() alignment = 64 : memref<32x256xf32, 1>
// CHECK:         memref.alloc() alignment = 64 : memref<256x32xf32, 1>
// CHECK:         memref.alloc() alignment = 64 : memref<32x32xf32, 1>
// CHECK:         memref.alloc() alignment = 64 : memref<32x32xf32, 1>
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
    // 5 hoisted L1 slots, slotCount = 1 each
    %bias_l1 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<32xf32>
    %a_l1 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<32x256xf32>
    %b_l1 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<256x32xf32>
    %acc_l1 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<32x32xf32>
    %out_l1 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<32x32xf32>
    %r:6 = scf.for %i = %c0 to %c128 step %c32
        iter_args(%out = %arg3, %sb = %bias_l1, %sa = %a_l1, %sw = %b_l1, %sacc = %acc_l1, %so = %out_l1)
        -> (tensor<128x64xf32>, tensor<32xf32>, tensor<32x256xf32>, tensor<256x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) {
      %rr:6 = scf.for %j = %c0 to %c64 step %c32
          iter_args(%out1 = %out, %sb1 = %sb, %sa1 = %sa, %sw1 = %sw, %sacc1 = %sacc, %so1 = %so)
          -> (tensor<128x64xf32>, tensor<32xf32>, tensor<32x256xf32>, tensor<256x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) {
        %cs = tensor.extract_slice %arg2[%i] [32] [1] : tensor<128xf32> to tensor<32xf32>
        %bias_v = bufferization.materialize_in_destination %cs in %sb1 : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
        %as = tensor.extract_slice %arg0[%i, 0] [32, 256] [1, 1] : tensor<128x256xf32> to tensor<32x256xf32>
        %a_v = bufferization.materialize_in_destination %as in %sa1 : (tensor<32x256xf32>, tensor<32x256xf32>) -> tensor<32x256xf32>
        %bs = tensor.extract_slice %arg1[0, %j] [256, 32] [1, 1] : tensor<256x64xf32> to tensor<256x32xf32>
        %b_v = bufferization.materialize_in_destination %bs in %sw1 : (tensor<256x32xf32>, tensor<256x32xf32>) -> tensor<256x32xf32>
        %f = linalg.fill ins(%cst : f32) outs(%sacc1 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %mm = linalg.matmul ins(%a_v, %b_v : tensor<32x256xf32>, tensor<256x32xf32>) outs(%f : tensor<32x32xf32>) -> tensor<32x32xf32>
        %g = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%bias_v, %mm : tensor<32xf32>, tensor<32x32xf32>) outs(%so1 : tensor<32x32xf32>) {
        ^bb0(%in: f32, %in_2: f32, %o: f32):
          %x = arith.addf %in_2, %in : f32
          %y = arith.maximumf %x, %cst : f32
          linalg.yield %y : f32
        } -> tensor<32x32xf32>
        %ins = tensor.insert_slice %g into %out1[%i, %j] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<128x64xf32>
        scf.yield %ins, %bias_v, %a_v, %b_v, %mm, %g : tensor<128x64xf32>, tensor<32xf32>, tensor<32x256xf32>, tensor<256x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>
      }
      scf.yield %rr#0, %rr#1, %rr#2, %rr#3, %rr#4, %rr#5 : tensor<128x64xf32>, tensor<32xf32>, tensor<32x256xf32>, tensor<256x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>
    }
    return %r#0 : tensor<128x64xf32>
  }
}
