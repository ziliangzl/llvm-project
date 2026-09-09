// RUN: mlir-opt %s --transform-interpreter | FileCheck %s --implicit-check-not="tensor<?"
// RUN: mlir-opt %s --transform-interpreter -one-shot-bufferize | FileCheck %s --check-prefix=BUF --implicit-check-not="memref<?"

// Positive case of SPM_TAIL_NORMALIZATION_EXAMPLE.md: same fully static,
// non-divisible input as tail-normalize-negative.mlir (130x64 * 64x200, tiles
// [64, 64]), but with NormalizeTileShapes strategy S1 (design 3.3) applied
// between TileForTier and MaterializeTensorStorage:
//
//   transform.loop.peel (innermost loop first, then the outer one)
//   + canonicalization/CSE      <-- mandatory second half, see the doc
//
// Peeling both dimensions yields 4 regions with 4 distinct static tile shapes:
//
//   main M x main N :  A 64x64  B 64x64  C 64x64   (the only 2-deep loop nest)
//   main M x tail N :  A 64x64  B 64x8   C 64x8
//   tail M x main N :  A 2x64   B 64x64  C 2x64
//   tail M x tail N :  A 2x64   B 64x8   C 2x8
//
// Every tile type and every alloc_tensor is static (--implicit-check-not
// asserts that no "tensor<?" survives anywhere in the function).

// CHECK-LABEL: func.func @matmul_130x64x200
// main M (0..128 step 64) x main N (0..192 step 64)
// CHECK:         scf.for %{{.*}} = %c0 to %c128 step %c64
// CHECK:           scf.for %{{.*}} = %c0 to %c192 step %c64
// CHECK:             tensor.extract_slice {{.*}} [64, 64] [1, 1] : tensor<130x64xf32> to tensor<64x64xf32>
// CHECK:             bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<64x64xf32>
// CHECK:             bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<64x64xf32>
// CHECK:             bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<64x64xf32>
// CHECK:             linalg.matmul ins({{.*}} : tensor<64x64xf32>, tensor<64x64xf32>) outs({{.*}} : tensor<64x64xf32>)
// main M x tail N (single trip, folded into the outer body)
// CHECK:           bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<64x8xf32>
// CHECK:           bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<64x64xf32>
// CHECK:           bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<64x8xf32>
// CHECK:           linalg.matmul ins({{.*}} : tensor<64x64xf32>, tensor<64x8xf32>) outs({{.*}} : tensor<64x8xf32>)
// tail M x main N
// CHECK:         scf.for
// CHECK:           bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<2x64xf32>
// CHECK:           bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<64x64xf32>
// CHECK:           bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<2x64xf32>
// CHECK:           linalg.matmul ins({{.*}} : tensor<2x64xf32>, tensor<64x64xf32>) outs({{.*}} : tensor<2x64xf32>)
// tail M x tail N
// CHECK:         bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<2x64xf32>
// CHECK:         bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<64x8xf32>
// CHECK:         bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<2x8xf32>
// CHECK:         linalg.matmul ins({{.*}} : tensor<2x64xf32>, tensor<64x8xf32>) outs({{.*}} : tensor<2x8xf32>)

// BUF-LABEL: func.func @matmul_130x64x200
// BUF-COUNT-12: memref.alloc() alignment = 64 : memref<{{.*}}, 1>
func.func @matmul_130x64x200(%A: tensor<130x64xf32>, %B: tensor<64x200xf32>,
                             %C: tensor<130x200xf32>) -> tensor<130x200xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<130x64xf32>, tensor<64x200xf32>)
                     outs(%C : tensor<130x200xf32>) -> tensor<130x200xf32>
  return %0 : tensor<130x200xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    // --- TileForTier ---
    %mm0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %tiled, %lm, %ln = transform.structured.tile_using_for %mm0 tile_sizes [64, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // --- NormalizeTileShapes, S1: peel innermost first, then outer ---
    %ln_for = transform.cast %ln : !transform.any_op to !transform.op<"scf.for">
    %ln_main, %ln_tail = transform.loop.peel %ln_for
      : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    %lm_for = transform.cast %lm : !transform.any_op to !transform.op<"scf.for">
    %lm_main, %lm_tail = transform.loop.peel %lm_for
      : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)

    // --- NormalizeTileShapes, mandatory second half: peeling only rewrites the
    // affine.min into an affine.apply; canonicalization is what turns the
    // dynamic tile types into static ones. ---
    %f = transform.structured.match ops{["func.func"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %f : !transform.any_op

    // --- MaterializeTensorStorage on every (main and tail) tile ---
    %mms = transform.structured.match ops{["linalg.matmul"]} in %f
      : (!transform.any_op) -> !transform.any_op
    %a = transform.get_operand %mms[0] : (!transform.any_op) -> !transform.any_value
    transform.structured.promote_tensor to 1 %a : !transform.any_value
    %b = transform.get_operand %mms[1] : (!transform.any_op) -> !transform.any_value
    transform.structured.promote_tensor to 1 %b : !transform.any_value
    %c = transform.get_operand %mms[2] : (!transform.any_op) -> !transform.any_value
    transform.structured.promote_tensor to 1 %c : !transform.any_value
    transform.yield
  }
}
