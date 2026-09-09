// RUN: mlir-opt %s --transform-interpreter -canonicalize -cse

// STEP 1 / 5 -- TileForTier: tile the matmul by [64, 64].
// BEFORE = the func below.  AFTER = run the RUN line (= step2-peel-inner.mlir's body).
//
// What to look at in the output: 130 % 64 != 0 and 200 % 64 != 0, so the tile
// extents cannot be constants. Tiling emits an affine.min per tiled dim and the
// tile types degrade to tensor<?x64xf32> / tensor<64x?xf32> / tensor<?x?xf32>.
func.func @matmul_130x64x200(%A: tensor<130x64xf32>, %B: tensor<64x200xf32>,
                             %C: tensor<130x200xf32>) -> tensor<130x200xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<130x64xf32>, tensor<64x200xf32>)
                     outs(%C : tensor<130x200xf32>) -> tensor<130x200xf32>
  return %0 : tensor<130x200xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %tiled, %lm, %ln = transform.structured.tile_using_for %mm tile_sizes [64, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
