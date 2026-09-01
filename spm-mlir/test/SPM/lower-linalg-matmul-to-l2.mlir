// RUN: spm-opt %s -spm-lower-linalg-matmul-to-l2 | FileCheck %s

// CHECK-LABEL: func.func @matmul_l2
// CHECK-SAME: (%[[A:.*]]: !spm.l2mem<64x32xf32>, %[[B:.*]]: !spm.l2mem<32x128xf32>, %[[C:.*]]: !spm.l2mem<64x128xf32>)
func.func @matmul_l2(%a: !spm.l2mem<64x32xf32>, %b: !spm.l2mem<32x128xf32>,
                      %c: !spm.l2mem<64x128xf32>) {
  // CHECK: %[[A_T:.*]] = spm.l2_load %[[A]] : <64x32xf32> -> tensor<64x32xf32>
  // CHECK: %[[B_T:.*]] = spm.l2_load %[[B]] : <32x128xf32> -> tensor<32x128xf32>
  // CHECK: %[[C_T:.*]] = spm.l2_load %[[C]] : <64x128xf32> -> tensor<64x128xf32>
  // CHECK: %[[R:.*]] = linalg.matmul ins(%[[A_T]], %[[B_T]] : tensor<64x32xf32>, tensor<32x128xf32>) outs(%[[C_T]] : tensor<64x128xf32>) -> tensor<64x128xf32>
  // CHECK: spm.l2_store %[[R]], %[[C]] : tensor<64x128xf32> -> <64x128xf32>
  linalg.matmul ins(%a, %b : !spm.l2mem<64x32xf32>, !spm.l2mem<32x128xf32>)
                outs(%c : !spm.l2mem<64x128xf32>)
  return
}

// A tensor-semantics matmul (the normal, upstream case) must be left alone.
// CHECK-LABEL: func.func @matmul_tensor_unaffected
func.func @matmul_tensor_unaffected(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>,
                                     %c: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<4x4xf32>, tensor<4x4xf32>) outs(%{{.*}} : tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK-NOT: spm.l2_load
  %r = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)
                      outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}
