// RUN: mlir-opt %s --transform-interpreter -canonicalize -cse | FileCheck %s
// RUN: mlir-opt %s --transform-interpreter -canonicalize -cse -one-shot-bufferize | FileCheck %s --check-prefix=BUF

// Negative case of SPM_TAIL_NORMALIZATION_EXAMPLE.md: fully static input
// (130x64 * 64x200), tiled [64, 64] with a non-divisible trip count and NO
// tail normalization. Both tile extents become affine.min values, so
// MaterializeTensorStorage (= transform.structured.promote_tensor) emits a
// DYNAMICALLY SIZED L1 allocation inside the loop nest.
//
// This is the exact form forbidden by design 3.3 / 7.2 V9 (alloc_tensor result
// type must be fully static) and by the spilling doc's invariant 10.

// CHECK-LABEL: func.func @matmul_130x64x200
// CHECK:         %[[M:.+]] = affine.min
// CHECK:         %[[N:.+]] = affine.min
// CHECK:         tensor.extract_slice {{.*}} to tensor<?x64xf32>
// CHECK:         bufferization.alloc_tensor(%[[M]]) <{memory_space = 1 : i64}> : tensor<?x64xf32>
// CHECK:         bufferization.alloc_tensor(%[[N]]) <{memory_space = 1 : i64}> : tensor<64x?xf32>
// CHECK:         bufferization.alloc_tensor(%[[M]], %[[N]]) <{memory_space = 1 : i64}> : tensor<?x?xf32>
// CHECK:         linalg.matmul ins({{.*}} : tensor<?x64xf32>, tensor<64x?xf32>) outs({{.*}} : tensor<?x?xf32>)

// BUF-LABEL: func.func @matmul_130x64x200
// BUF:         scf.for
// BUF:           scf.for
// BUF:             memref.alloc(%{{.+}}) alignment = 64 : memref<?x64xf32, 1>
// BUF:             memref.alloc(%{{.+}}) alignment = 64 : memref<64x?xf32, 1>
// BUF:             memref.alloc(%{{.+}}, %{{.+}}) alignment = 64 : memref<?x?xf32, 1>
func.func @matmul_130x64x200(%A: tensor<130x64xf32>, %B: tensor<64x200xf32>,
                             %C: tensor<130x200xf32>) -> tensor<130x200xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<130x64xf32>, tensor<64x200xf32>)
                     outs(%C : tensor<130x200xf32>) -> tensor<130x200xf32>
  return %0 : tensor<130x200xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    // TileForTier only -- NormalizeTileShapes deliberately skipped.
    %mm0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %tiled, %lm, %ln = transform.structured.tile_using_for %mm0 tile_sizes [64, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    // MaterializeTensorStorage on the (dynamic) tiles.
    %a = transform.get_operand %tiled[0] : (!transform.any_op) -> !transform.any_value
    transform.structured.promote_tensor to 1 %a : !transform.any_value
    %b = transform.get_operand %tiled[1] : (!transform.any_op) -> !transform.any_value
    transform.structured.promote_tensor to 1 %b : !transform.any_value
    %c = transform.get_operand %tiled[2] : (!transform.any_op) -> !transform.any_value
    transform.structured.promote_tensor to 1 %c : !transform.any_value
    transform.yield
  }
}
