// RUN: mlir-opt %s --transform-interpreter | FileCheck %s

// STEP 3 / 5 -- NormalizeTileShapes, peel the OUTER (M) loop.
// BEFORE = the func below (= step 2's output).  AFTER = run the RUN line.
//
// The M loop [0, 130) step 64 is split into:
//     main      [0, 128) step 64    (2 full row-tiles)
//     remainder [128, 130) step 64  (the 2 leftover rows)
// Peeling the outer loop CLONES its whole body, so the two inner loops that
// step 2 produced are duplicated: 2 x 2 = 4 regions, which is why two
// non-divisible dimensions cost 2^2 copies of the compute code.
//
// That is also why the order matters: peel inner first, then outer, and two
// transform.loop.peel calls are enough. Peeling outer first would leave two
// separate inner loops to find and peel afterwards.
//
// After this step every affine.min is gone (each region knows its own extent),
// but the tile TYPES are still tensor<?x?xf32> -- see step 4.

// CHECK-LABEL: func.func @matmul_130x64x200
// CHECK-NOT:     affine.min
// main M x main N
// CHECK:         scf.for %{{.*}} = %c0 to %c128 step %c64
// CHECK:           scf.for %{{.*}} = %c0 to %c192 step %c64
// main M x tail N
// CHECK:           scf.for %{{.*}} = %c192 to %c200 step %c64
// tail M x main N / tail M x tail N
// CHECK:         scf.for %{{.*}} = %c128 to %c130 step %c64
// CHECK:           scf.for %{{.*}} = %c0 to %c192 step %c64
// CHECK:           scf.for %{{.*}} = %c192 to %c200 step %c64

#map = affine_map<(d0) -> (-d0 + 130, 64)>
#map1 = affine_map<() -> (64)>
#map2 = affine_map<(d0) -> (-d0 + 200)>
func.func @matmul_130x64x200(%arg0: tensor<130x64xf32>, %arg1: tensor<64x200xf32>, %arg2: tensor<130x200xf32>) -> tensor<130x200xf32> {
  %c0 = arith.constant 0 : index
  %c130 = arith.constant 130 : index
  %c200 = arith.constant 200 : index
  %c64 = arith.constant 64 : index
  %0 = scf.for %arg3 = %c0 to %c130 step %c64 iter_args(%arg4 = %arg2) -> (tensor<130x200xf32>) {
    %c192 = arith.constant 192 : index
    %1 = scf.for %arg5 = %c0 to %c192 step %c64 iter_args(%arg6 = %arg4) -> (tensor<130x200xf32>) {
      %3 = affine.min #map(%arg3)
      %4 = affine.apply #map1()
      %extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [%3, 64] [1, 1] : tensor<130x64xf32> to tensor<?x64xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg5] [64, %4] [1, 1] : tensor<64x200xf32> to tensor<64x?xf32>
      %extracted_slice_1 = tensor.extract_slice %arg6[%arg3, %arg5] [%3, %4] [1, 1] : tensor<130x200xf32> to tensor<?x?xf32>
      %5 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<?x64xf32>, tensor<64x?xf32>) outs(%extracted_slice_1 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %inserted_slice = tensor.insert_slice %5 into %arg6[%arg3, %arg5] [%3, %4] [1, 1] : tensor<?x?xf32> into tensor<130x200xf32>
      scf.yield %inserted_slice : tensor<130x200xf32>
    }
    %2 = scf.for %arg5 = %c192 to %c200 step %c64 iter_args(%arg6 = %1) -> (tensor<130x200xf32>) {
      %3 = affine.min #map(%arg3)
      %4 = affine.apply #map2(%arg5)
      %extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [%3, 64] [1, 1] : tensor<130x64xf32> to tensor<?x64xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg5] [64, %4] [1, 1] : tensor<64x200xf32> to tensor<64x?xf32>
      %extracted_slice_1 = tensor.extract_slice %arg6[%arg3, %arg5] [%3, %4] [1, 1] : tensor<130x200xf32> to tensor<?x?xf32>
      %5 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<?x64xf32>, tensor<64x?xf32>) outs(%extracted_slice_1 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %inserted_slice = tensor.insert_slice %5 into %arg6[%arg3, %arg5] [%3, %4] [1, 1] : tensor<?x?xf32> into tensor<130x200xf32>
      scf.yield %inserted_slice : tensor<130x200xf32>
    }
    scf.yield %2 : tensor<130x200xf32>
  }
  return %0 : tensor<130x200xf32>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    // outer scf.for = the 2nd scf.for ancestor of the tiled matmuls
    %mms = transform.structured.match ops{["linalg.matmul"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %outer = transform.get_parent_op %mms {op_name = "scf.for", nth_parent = 2, deduplicate}
      : (!transform.any_op) -> !transform.any_op
    %outer_for = transform.cast %outer : !transform.any_op to !transform.op<"scf.for">
    %m_main, %m_tail = transform.loop.peel %outer_for
      : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
