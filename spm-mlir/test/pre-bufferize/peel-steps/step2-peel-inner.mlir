// RUN: mlir-opt %s --transform-interpreter | FileCheck %s

// STEP 2 / 5 -- NormalizeTileShapes, peel the INNER (N) loop.
// BEFORE = the func below (= step 1's output).  AFTER = run the RUN line.
//
// The N loop [0, 200) step 64 is split into:
//     main      [0, 192) step 64    (192 = 200 - (200 mod 64): 3 full tiles)
//     remainder [192, 200) step 64  (the 8 leftover columns)
// Both copies stay inside the still-unpeeled M loop, so the function now has
// 1 outer loop containing 2 inner loops.
//
// The affine.min on the N extent is resolved separately in each copy:
//     main      -> affine.apply () -> (64)
//     remainder -> affine.apply (d0) -> (-d0 + 200)
// The affine.min on the M extent is untouched (that is step 3), and the types
// are still tensor<?x...> (that is step 4).

// CHECK-DAG:   #[[$M:.+]] = affine_map<(d0) -> (-d0 + 130, 64)>
// CHECK-DAG:   #[[$FULL:.+]] = affine_map<() -> (64)>
// CHECK-DAG:   #[[$REST:.+]] = affine_map<(d0) -> (-d0 + 200)>
// CHECK-LABEL: func.func @matmul_130x64x200
// CHECK:         scf.for %{{.*}} = %c0 to %c130 step %c64
// CHECK:           %[[C192:.+]] = arith.constant 192 : index
// CHECK:           %[[NMAIN:.+]] = scf.for %{{.*}} = %c0 to %[[C192]] step %c64
// CHECK:             affine.min #[[$M]]
// CHECK:             affine.apply #[[$FULL]]()
// CHECK:           scf.for %{{.*}} = %[[C192]] to %c200 step %c64 iter_args(%{{.*}} = %[[NMAIN]])
// CHECK:             affine.min #[[$M]]
// CHECK:             affine.apply #[[$REST]]

#map = affine_map<(d0) -> (-d0 + 130, 64)>
#map1 = affine_map<(d0) -> (-d0 + 200, 64)>
func.func @matmul_130x64x200(%arg0: tensor<130x64xf32>, %arg1: tensor<64x200xf32>, %arg2: tensor<130x200xf32>) -> tensor<130x200xf32> {
  %c0 = arith.constant 0 : index
  %c130 = arith.constant 130 : index
  %c200 = arith.constant 200 : index
  %c64 = arith.constant 64 : index
  %0 = scf.for %arg3 = %c0 to %c130 step %c64 iter_args(%arg4 = %arg2) -> (tensor<130x200xf32>) {
    %1 = scf.for %arg5 = %c0 to %c200 step %c64 iter_args(%arg6 = %arg4) -> (tensor<130x200xf32>) {
      %2 = affine.min #map(%arg3)
      %3 = affine.min #map1(%arg5)
      %extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [%2, 64] [1, 1] : tensor<130x64xf32> to tensor<?x64xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg5] [64, %3] [1, 1] : tensor<64x200xf32> to tensor<64x?xf32>
      %extracted_slice_1 = tensor.extract_slice %arg6[%arg3, %arg5] [%2, %3] [1, 1] : tensor<130x200xf32> to tensor<?x?xf32>
      %4 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<?x64xf32>, tensor<64x?xf32>) outs(%extracted_slice_1 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %inserted_slice = tensor.insert_slice %4 into %arg6[%arg3, %arg5] [%2, %3] [1, 1] : tensor<?x?xf32> into tensor<130x200xf32>
      scf.yield %inserted_slice : tensor<130x200xf32>
    }
    scf.yield %1 : tensor<130x200xf32>
  }
  return %0 : tensor<130x200xf32>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    // innermost scf.for = the direct scf.for parent of the tiled matmul
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %inner = transform.get_parent_op %mm {op_name = "scf.for", deduplicate}
      : (!transform.any_op) -> !transform.any_op
    %inner_for = transform.cast %inner : !transform.any_op to !transform.op<"scf.for">
    %n_main, %n_tail = transform.loop.peel %inner_for
      : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
