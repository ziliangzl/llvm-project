// RUN: mlir-opt %s -scf-for-loop-peeling=skip-partial=false -canonicalize -cse | FileCheck %s --implicit-check-not="tensor<?"
// RUN: mlir-opt %s -scf-for-loop-peeling -canonicalize -cse | FileCheck %s --check-prefix=SKIPPARTIAL

// Same case as tail-normalize-peel.mlir, but driven by the stock upstream PASS
// (-scf-for-loop-peeling) instead of transform.loop.peel. Input is the
// already-tiled IR (130x64 * 64x200, tiles [64, 64], both dims non-divisible).
//
// With skip-partial=false the output is byte-identical to the transform-op
// route: 4 regions, every tile static.
//
// With the DEFAULT skip-partial=true the loops nested inside a partial (tail)
// iteration are deliberately NOT peeled ("partial iterations are not usually
// performance critical", LoopSpecialization.cpp:329-337), so the M-tail region
// keeps its affine.min and its DYNAMIC tile types -> V9 still violated there.
// => NormalizeTileShapes must pass skip-partial=false.

// CHECK-LABEL: func.func @matmul_130x64x200
// CHECK:         scf.for %{{.*}} = %c0 to %c128 step %c64
// CHECK:           scf.for %{{.*}} = %c0 to %c192 step %c64
// CHECK:             linalg.matmul ins({{.*}} : tensor<64x64xf32>, tensor<64x64xf32>) outs({{.*}} : tensor<64x64xf32>)
// CHECK:           linalg.matmul ins({{.*}} : tensor<64x64xf32>, tensor<64x8xf32>) outs({{.*}} : tensor<64x8xf32>)
// CHECK:         scf.for %{{.*}} = %c0 to %c192 step %c64
// CHECK:           linalg.matmul ins({{.*}} : tensor<2x64xf32>, tensor<64x64xf32>) outs({{.*}} : tensor<2x64xf32>)
// CHECK:         linalg.matmul ins({{.*}} : tensor<2x64xf32>, tensor<64x8xf32>) outs({{.*}} : tensor<2x8xf32>)

// The main M x main/tail N regions are fine ...
// SKIPPARTIAL-LABEL: func.func @matmul_130x64x200
// SKIPPARTIAL:         scf.for %{{.*}} = %c0 to %c128 step %c64
// SKIPPARTIAL:           scf.for %{{.*}} = %c0 to %c192 step %c64
// SKIPPARTIAL:             linalg.matmul ins({{.*}} : tensor<64x64xf32>, tensor<64x64xf32>) outs({{.*}} : tensor<64x64xf32>)
// SKIPPARTIAL:           linalg.matmul ins({{.*}} : tensor<64x64xf32>, tensor<64x8xf32>) outs({{.*}} : tensor<64x8xf32>)
// ... but the M-tail region is left unpeeled and stays dynamic:
// SKIPPARTIAL:         scf.for %{{.*}} = %c0 to %c200 step %c64
// SKIPPARTIAL:           affine.min
// SKIPPARTIAL:           tensor.extract_slice {{.*}} to tensor<64x?xf32>
// SKIPPARTIAL:           linalg.matmul ins({{.*}} : tensor<2x64xf32>, tensor<64x?xf32>) outs({{.*}} : tensor<2x?xf32>)

#map = affine_map<(d0) -> (-d0 + 130, 64)>
#map1 = affine_map<(d0) -> (-d0 + 200, 64)>
module {
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
}


