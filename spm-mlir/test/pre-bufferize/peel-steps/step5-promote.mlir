// RUN: mlir-opt %s --transform-interpreter | FileCheck %s --implicit-check-not="tensor<?"

// STEP 5 / 5 -- MaterializeTensorStorage: promote every tile into L1 (space 1).
// BEFORE = the func below (= step 4's output).  AFTER = run the RUN line.
//
// This is the step that tail normalization exists for. promote_tensor inserts
//   %s = bufferization.alloc_tensor() <{memory_space = 1}> : <tile type>
//   %v = bufferization.materialize_in_destination %tile in %s   (if the tile is read)
// so the tile TYPE becomes the allocation size. Because step 4 made every tile
// type static, all 12 allocations here are static too -- compare with
// ../tail-normalize-negative.mlir, where skipping steps 2-4 gives
// alloc_tensor(%2, %3) : tensor<?x?xf32>, a dynamically sized SRAM allocation.
//
// 4 regions x 3 operands = 12 allocations, in 4 distinct shape groups:
//   A 64x64 / 64x64 / 64x64      B 64x64 / 64x8 / 64x8
//   C 2x64  / 64x64 / 2x64       D 2x64  / 64x8 / 2x8
// Their live ranges do not overlap, so the arena planner can reuse offsets
// across regions -- but it must see 4 groups, not 1.

// CHECK-LABEL: func.func @matmul_130x64x200
// CHECK:         scf.for %{{.*}} = %c0 to %c128 step %c64
// CHECK:           scf.for %{{.*}} = %c0 to %c192 step %c64
// CHECK-COUNT-3:     bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<64x64xf32>
// CHECK:             linalg.matmul
// CHECK:           bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<64x64xf32>
// CHECK:           bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<64x8xf32>
// CHECK:           bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<64x8xf32>
// CHECK:           linalg.matmul ins({{.*}} : tensor<64x64xf32>, tensor<64x8xf32>) outs({{.*}} : tensor<64x8xf32>)
// CHECK:         scf.for %{{.*}} = %c0 to %c192 step %c64
// CHECK:           bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<2x64xf32>
// CHECK:           bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<64x64xf32>
// CHECK:           bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<2x64xf32>
// CHECK:         bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<2x64xf32>
// CHECK:         bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<64x8xf32>
// CHECK:         bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<2x8xf32>

func.func @matmul_130x64x200(%arg0: tensor<130x64xf32>, %arg1: tensor<64x200xf32>, %arg2: tensor<130x200xf32>) -> tensor<130x200xf32> {
  %c192 = arith.constant 192 : index
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %0 = scf.for %arg3 = %c0 to %c128 step %c64 iter_args(%arg4 = %arg2) -> (tensor<130x200xf32>) {
    %3 = scf.for %arg5 = %c0 to %c192 step %c64 iter_args(%arg6 = %arg4) -> (tensor<130x200xf32>) {
      %extracted_slice_6 = tensor.extract_slice %arg0[%arg3, 0] [64, 64] [1, 1] : tensor<130x64xf32> to tensor<64x64xf32>
      %extracted_slice_7 = tensor.extract_slice %arg1[0, %arg5] [64, 64] [1, 1] : tensor<64x200xf32> to tensor<64x64xf32>
      %extracted_slice_8 = tensor.extract_slice %arg6[%arg3, %arg5] [64, 64] [1, 1] : tensor<130x200xf32> to tensor<64x64xf32>
      %5 = linalg.matmul ins(%extracted_slice_6, %extracted_slice_7 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%extracted_slice_8 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %inserted_slice_9 = tensor.insert_slice %5 into %arg6[%arg3, %arg5] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<130x200xf32>
      scf.yield %inserted_slice_9 : tensor<130x200xf32>
    }
    %extracted_slice_2 = tensor.extract_slice %arg0[%arg3, 0] [64, 64] [1, 1] : tensor<130x64xf32> to tensor<64x64xf32>
    %extracted_slice_3 = tensor.extract_slice %arg1[0, 192] [64, 8] [1, 1] : tensor<64x200xf32> to tensor<64x8xf32>
    %extracted_slice_4 = tensor.extract_slice %3[%arg3, 192] [64, 8] [1, 1] : tensor<130x200xf32> to tensor<64x8xf32>
    %4 = linalg.matmul ins(%extracted_slice_2, %extracted_slice_3 : tensor<64x64xf32>, tensor<64x8xf32>) outs(%extracted_slice_4 : tensor<64x8xf32>) -> tensor<64x8xf32>
    %inserted_slice_5 = tensor.insert_slice %4 into %3[%arg3, 192] [64, 8] [1, 1] : tensor<64x8xf32> into tensor<130x200xf32>
    scf.yield %inserted_slice_5 : tensor<130x200xf32>
  }
  %1 = scf.for %arg3 = %c0 to %c192 step %c64 iter_args(%arg4 = %0) -> (tensor<130x200xf32>) {
    %extracted_slice_2 = tensor.extract_slice %arg0[128, 0] [2, 64] [1, 1] : tensor<130x64xf32> to tensor<2x64xf32>
    %extracted_slice_3 = tensor.extract_slice %arg1[0, %arg3] [64, 64] [1, 1] : tensor<64x200xf32> to tensor<64x64xf32>
    %extracted_slice_4 = tensor.extract_slice %arg4[128, %arg3] [2, 64] [1, 1] : tensor<130x200xf32> to tensor<2x64xf32>
    %3 = linalg.matmul ins(%extracted_slice_2, %extracted_slice_3 : tensor<2x64xf32>, tensor<64x64xf32>) outs(%extracted_slice_4 : tensor<2x64xf32>) -> tensor<2x64xf32>
    %inserted_slice_5 = tensor.insert_slice %3 into %arg4[128, %arg3] [2, 64] [1, 1] : tensor<2x64xf32> into tensor<130x200xf32>
    scf.yield %inserted_slice_5 : tensor<130x200xf32>
  }
  %extracted_slice = tensor.extract_slice %arg0[128, 0] [2, 64] [1, 1] : tensor<130x64xf32> to tensor<2x64xf32>
  %extracted_slice_0 = tensor.extract_slice %arg1[0, 192] [64, 8] [1, 1] : tensor<64x200xf32> to tensor<64x8xf32>
  %extracted_slice_1 = tensor.extract_slice %1[128, 192] [2, 8] [1, 1] : tensor<130x200xf32> to tensor<2x8xf32>
  %2 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<2x64xf32>, tensor<64x8xf32>) outs(%extracted_slice_1 : tensor<2x8xf32>) -> tensor<2x8xf32>
  %inserted_slice = tensor.insert_slice %2 into %1[128, 192] [2, 8] [1, 1] : tensor<2x8xf32> into tensor<130x200xf32>
  return %inserted_slice : tensor<130x200xf32>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    // one handle holding all 4 matmuls (main/tail x main/tail)
    %mms = transform.structured.match ops{["linalg.matmul"]} in %arg1
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
