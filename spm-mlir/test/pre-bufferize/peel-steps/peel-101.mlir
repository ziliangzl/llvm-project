// RUN: mlir-opt %s --transform-interpreter | FileCheck %s
// RUN: mlir-opt %s --transform-interpreter -canonicalize -cse | FileCheck %s --check-prefix=CANON

// peel 101 -- what transform.loop.peel actually does, with no linalg noise.
//
// The loop below is a 1-D miniature of what tiling produces: trip range
// [0, 130) with step 64, and an affine.min that clamps the last iteration's
// extent to what is left (130 - 128 = 2). "%n" is therefore 64, 64, 2.
//
// peel rewrites ONE loop into TWO:
//   main loop      [0, 128)   step 64   <- 128 = 130 - (130 - 0) mod 64
//   remainder loop [128, 130) step 64   <- exactly the leftover iterations
// and then rewrites the affine.min inside each of them:
//   in the main loop      it is provably the full step  -> affine.apply () -> (64)
//   in the remainder loop it is provably ub - iv        -> affine.apply (d0) -> (-d0 + 130)
// The op count grows, but every extent becomes a closed-form expression.

// --- after peel only: two loops, extents are affine.apply, types still dynamic
// CHECK-DAG:   #[[$FULL:.+]] = affine_map<() -> (64)>
// CHECK-DAG:   #[[$REST:.+]] = affine_map<(d0) -> (-d0 + 130)>
// CHECK-LABEL: func.func @peel_101
// CHECK:         %[[C128:.+]] = arith.constant 128 : index
// CHECK:         %[[MAIN:.+]] = scf.for %{{.*}} = %c0 to %[[C128]] step %c64
// CHECK:           affine.apply #[[$FULL]]()
// CHECK:           tensor.extract_slice {{.*}} : tensor<130xf32> to tensor<?xf32>
// CHECK:         scf.for %{{.*}} = %[[C128]] to %c130 step %c64 iter_args(%{{.*}} = %[[MAIN]])
// CHECK:           affine.apply #[[$REST]](%{{.*}})
// CHECK:           tensor.extract_slice {{.*}} : tensor<130xf32> to tensor<?xf32>

// --- after peel + canonicalize: extents are constants, types are static,
//     and the 1-trip remainder loop is folded into straight-line code
// CANON-LABEL: func.func @peel_101
// CANON:         %[[R:.+]] = scf.for %{{.*}} = %c0 to %c128 step %c64
// CANON:           tensor.extract_slice %{{.*}}[%{{.*}}] [64] [1] : tensor<130xf32> to tensor<64xf32>
// CANON:           linalg.fill {{.*}} -> tensor<64xf32>
// CANON:         }
// CANON-NOT:     scf.for
// CANON:         tensor.extract_slice %[[R]][128] [2] [1] : tensor<130xf32> to tensor<2xf32>
// CANON:         linalg.fill {{.*}} -> tensor<2xf32>

func.func @peel_101(%t: tensor<130xf32>) -> tensor<130xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c130 = arith.constant 130 : index
  %r = scf.for %i = %c0 to %c130 step %c64 iter_args(%acc = %t) -> (tensor<130xf32>) {
    %n = affine.min affine_map<(d0) -> (-d0 + 130, 64)>(%i)
    %s = tensor.extract_slice %acc[%i] [%n] [1] : tensor<130xf32> to tensor<?xf32>
    %f = linalg.fill ins(%cst : f32) outs(%s : tensor<?xf32>) -> tensor<?xf32>
    %ins = tensor.insert_slice %f into %acc[%i] [%n] [1] : tensor<?xf32> into tensor<130xf32>
    scf.yield %ins : tensor<130xf32>
  }
  return %r : tensor<130xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %l = transform.structured.match ops{["scf.for"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %l_for = transform.cast %l : !transform.any_op to !transform.op<"scf.for">
    %main, %tail = transform.loop.peel %l_for
      : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
