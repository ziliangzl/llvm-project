// RUN: mlir-opt %s --transform-interpreter -verify-diagnostics

// Idempotence trap recorded in SPM_TAIL_NORMALIZATION_EXAMPLE.md:
// on an already evenly divisible loop (128x64 * 64x192, tiles [64, 64]),
// scf::peelForLoop bails out (LoopSpecialization.cpp:136-138, "no
// specialization necessary if step already divides upper bound evenly") and
// transform.loop.peel turns that into a silenceable error -- the
// fail_if_already_divisible attribute is NOT consulted by the implementation
// (SCFTransformOps.cpp:274-282).
//
// => NormalizeTileShapes must test (ub - lb) % step == 0 itself and skip, or
//    call mlir::linalg::peelLoops(), which swallows the failure.
func.func @matmul_divisible(%A: tensor<128x64xf32>, %B: tensor<64x192xf32>,
                            %C: tensor<128x192xf32>) -> tensor<128x192xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<128x64xf32>, tensor<64x192xf32>)
                     outs(%C : tensor<128x192xf32>) -> tensor<128x192xf32>
  return %0 : tensor<128x192xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %mm0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %tiled, %lm, %ln = transform.structured.tile_using_for %mm0 tile_sizes [64, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %ln_for = transform.cast %ln : !transform.any_op to !transform.op<"scf.for">
    // expected-error @below {{failed to peel the last iteration}}
    %ln_main, %ln_tail = transform.loop.peel %ln_for
      : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
