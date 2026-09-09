// RUN: mlir-opt %s --transform-interpreter -canonicalize -cse | FileCheck %s

// Step 0 of the HoistOnly example (SPM_HOIST_ONLY_EXAMPLE.md):
// how the "before" IR is produced from an upstream-style linalg input.
//
// Op structure taken from mlir/test/Dialect/Linalg/transform-tile-and-fuse.mlir
// (fill -> matmul -> generic), but with fully static shapes and scf.for loops
// (SPM_PRE_BUFFERIZE_DESIGN.md 3.7).
//
// TileForTier            = transform.structured.fuse (tile + fuse producers)
// MaterializeTensorStorage = transform.structured.promote_tensor to 1 (= L1)
//
// Result: 5 bufferization.alloc_tensor INSIDE the inner loop body, i.e. the
// form that design 5.8/Q1 and 7.2/V10 forbid.

// CHECK-LABEL: func.func @fused_matmul_relu
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK-COUNT-5:     bufferization.alloc_tensor() <{memory_space = 1 : i64}>
// CHECK:             tensor.insert_slice
// CHECK:             scf.yield
func.func @fused_matmul_relu(%A: tensor<128x256xf32>, %B: tensor<256x64xf32>,
                             %C: tensor<128xf32>, %D: tensor<128x64xf32>)
    -> tensor<128x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill {__producer__} ins(%cst : f32)
       outs(%D : tensor<128x64xf32>) -> tensor<128x64xf32>
  %1 = linalg.matmul {__producer__}
       ins(%A, %B : tensor<128x256xf32>, tensor<256x64xf32>)
       outs(%0 : tensor<128x64xf32>) -> tensor<128x64xf32>
  %2 = linalg.generic {
        __root__,
        indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                         affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
       ins(%C, %1 : tensor<128xf32>, tensor<128x64xf32>)
       outs(%D : tensor<128x64xf32>) {
  ^bb0(%b: f32, %x: f32, %o: f32):
    %m = arith.addf %x, %b : f32
    %r = arith.maximumf %m, %cst : f32
    linalg.yield %r : f32
  } -> tensor<128x64xf32>
  return %2 : tensor<128x64xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %root = transform.structured.match attributes{"__root__"} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %fused, %lm, %ln = transform.structured.fuse %root tile_sizes [32, 32] {apply_cleanup}
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // NOTE: the matches below MUST be scoped to the generated loop (%ln), not to
    // the whole payload: tile-and-fuse leaves the original untiled fill/matmul
    // behind as dead ops until canonicalization runs, and promoting *those*
    // silently promotes the whole DRAM tensor instead of the tile.
    %mm = transform.structured.match ops{["linalg.matmul"]} in %ln
      : (!transform.any_op) -> !transform.any_op
    %fl = transform.structured.match ops{["linalg.fill"]} in %ln
      : (!transform.any_op) -> !transform.any_op

    %a_tile = transform.get_operand %mm[0] : (!transform.any_op) -> !transform.any_value
    transform.structured.promote_tensor to 1 %a_tile : !transform.any_value
    %b_tile = transform.get_operand %mm[1] : (!transform.any_op) -> !transform.any_value
    transform.structured.promote_tensor to 1 %b_tile : !transform.any_value
    %acc_tile = transform.get_operand %fl[1] : (!transform.any_op) -> !transform.any_value
    transform.structured.promote_tensor to 1 %acc_tile : !transform.any_value
    %bias_tile = transform.get_operand %fused[0] : (!transform.any_op) -> !transform.any_value
    transform.structured.promote_tensor to 1 %bias_tile : !transform.any_value
    %out_tile = transform.get_operand %fused[2] : (!transform.any_op) -> !transform.any_value
    transform.structured.promote_tensor to 1 %out_tile : !transform.any_value
    transform.yield
  }
}
