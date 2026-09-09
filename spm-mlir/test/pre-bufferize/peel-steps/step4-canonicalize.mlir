// RUN: mlir-opt %s -canonicalize -cse | FileCheck %s --implicit-check-not="tensor<?"

// STEP 4 / 5 -- NormalizeTileShapes, the mandatory second half: canonicalize.
// BEFORE = the func below (= step 3's output).  AFTER = run the RUN line.
// NOTE: no transform script here, this step is plain -canonicalize -cse.
//
// Peeling alone left every extent as an affine.apply and every tile type as
// tensor<?x?xf32>. Canonicalization is what makes them static:
//   1. affine.apply with constant operands folds to an arith.constant
//   2. tensor.extract_slice / insert_slice with constant sizes fold their
//      result type to a static one, which propagates into linalg.matmul
//   3. a loop whose trip count is 1 is promoted (its body is inlined), which
//      is why the N-tail and M-tail loops disappear as loops here
//
// This is the step that satisfies design 7.2 V9 (alloc_tensor must be static),
// so NormalizeTileShapes must run it itself -- not leave it to a later pass.

// CHECK-LABEL: func.func @matmul_130x64x200
// region A: main M x main N -- the only remaining 2-deep nest
// CHECK:         scf.for %{{.*}} = %c0 to %c128 step %c64
// CHECK:           scf.for %{{.*}} = %c0 to %c192 step %c64
// CHECK:             linalg.matmul ins({{.*}} : tensor<64x64xf32>, tensor<64x64xf32>) outs({{.*}} : tensor<64x64xf32>)
// region B: main M x tail N -- 1 trip, inlined into the M loop body
// CHECK:           linalg.matmul ins({{.*}} : tensor<64x64xf32>, tensor<64x8xf32>) outs({{.*}} : tensor<64x8xf32>)
// region C: tail M x main N -- the M loop had 1 trip, so only the N loop is left
// CHECK:         scf.for %{{.*}} = %c0 to %c192 step %c64
// CHECK:           linalg.matmul ins({{.*}} : tensor<2x64xf32>, tensor<64x64xf32>) outs({{.*}} : tensor<2x64xf32>)
// region D: tail M x tail N -- straight-line code
// CHECK:         linalg.matmul ins({{.*}} : tensor<2x64xf32>, tensor<64x8xf32>) outs({{.*}} : tensor<2x8xf32>)

#map = affine_map<() -> (64)>
#map1 = affine_map<(d0) -> (-d0 + 200)>
#map2 = affine_map<(d0) -> (-d0 + 130)>
func.func @matmul_130x64x200(%arg0: tensor<130x64xf32>, %arg1: tensor<64x200xf32>, %arg2: tensor<130x200xf32>) -> tensor<130x200xf32> {
  %c0 = arith.constant 0 : index
  %c130 = arith.constant 130 : index
  %c200 = arith.constant 200 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %0 = scf.for %arg3 = %c0 to %c128 step %c64 iter_args(%arg4 = %arg2) -> (tensor<130x200xf32>) {
    %c192 = arith.constant 192 : index
    %2 = scf.for %arg5 = %c0 to %c192 step %c64 iter_args(%arg6 = %arg4) -> (tensor<130x200xf32>) {
      %4 = affine.apply #map()
      %5 = affine.apply #map()
      %extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [%4, 64] [1, 1] : tensor<130x64xf32> to tensor<?x64xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg5] [64, %5] [1, 1] : tensor<64x200xf32> to tensor<64x?xf32>
      %extracted_slice_1 = tensor.extract_slice %arg6[%arg3, %arg5] [%4, %5] [1, 1] : tensor<130x200xf32> to tensor<?x?xf32>
      %6 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<?x64xf32>, tensor<64x?xf32>) outs(%extracted_slice_1 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %inserted_slice = tensor.insert_slice %6 into %arg6[%arg3, %arg5] [%4, %5] [1, 1] : tensor<?x?xf32> into tensor<130x200xf32>
      scf.yield %inserted_slice : tensor<130x200xf32>
    }
    %3 = scf.for %arg5 = %c192 to %c200 step %c64 iter_args(%arg6 = %2) -> (tensor<130x200xf32>) {
      %4 = affine.apply #map()
      %5 = affine.apply #map1(%arg5)
      %extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [%4, 64] [1, 1] : tensor<130x64xf32> to tensor<?x64xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg5] [64, %5] [1, 1] : tensor<64x200xf32> to tensor<64x?xf32>
      %extracted_slice_1 = tensor.extract_slice %arg6[%arg3, %arg5] [%4, %5] [1, 1] : tensor<130x200xf32> to tensor<?x?xf32>
      %6 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<?x64xf32>, tensor<64x?xf32>) outs(%extracted_slice_1 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %inserted_slice = tensor.insert_slice %6 into %arg6[%arg3, %arg5] [%4, %5] [1, 1] : tensor<?x?xf32> into tensor<130x200xf32>
      scf.yield %inserted_slice : tensor<130x200xf32>
    }
    scf.yield %3 : tensor<130x200xf32>
  }
  %1 = scf.for %arg3 = %c128 to %c130 step %c64 iter_args(%arg4 = %0) -> (tensor<130x200xf32>) {
    %c192 = arith.constant 192 : index
    %2 = scf.for %arg5 = %c0 to %c192 step %c64 iter_args(%arg6 = %arg4) -> (tensor<130x200xf32>) {
      %4 = affine.apply #map2(%arg3)
      %5 = affine.apply #map()
      %extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [%4, 64] [1, 1] : tensor<130x64xf32> to tensor<?x64xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg5] [64, %5] [1, 1] : tensor<64x200xf32> to tensor<64x?xf32>
      %extracted_slice_1 = tensor.extract_slice %arg6[%arg3, %arg5] [%4, %5] [1, 1] : tensor<130x200xf32> to tensor<?x?xf32>
      %6 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<?x64xf32>, tensor<64x?xf32>) outs(%extracted_slice_1 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %inserted_slice = tensor.insert_slice %6 into %arg6[%arg3, %arg5] [%4, %5] [1, 1] : tensor<?x?xf32> into tensor<130x200xf32>
      scf.yield %inserted_slice : tensor<130x200xf32>
    }
    %3 = scf.for %arg5 = %c192 to %c200 step %c64 iter_args(%arg6 = %2) -> (tensor<130x200xf32>) {
      %4 = affine.apply #map2(%arg3)
      %5 = affine.apply #map1(%arg5)
      %extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [%4, 64] [1, 1] : tensor<130x64xf32> to tensor<?x64xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg5] [64, %5] [1, 1] : tensor<64x200xf32> to tensor<64x?xf32>
      %extracted_slice_1 = tensor.extract_slice %arg6[%arg3, %arg5] [%4, %5] [1, 1] : tensor<130x200xf32> to tensor<?x?xf32>
      %6 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<?x64xf32>, tensor<64x?xf32>) outs(%extracted_slice_1 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %inserted_slice = tensor.insert_slice %6 into %arg6[%arg3, %arg5] [%4, %5] [1, 1] : tensor<?x?xf32> into tensor<130x200xf32>
      scf.yield %inserted_slice : tensor<130x200xf32>
    }
    scf.yield %3 : tensor<130x200xf32>
  }
  return %1 : tensor<130x200xf32>
}

