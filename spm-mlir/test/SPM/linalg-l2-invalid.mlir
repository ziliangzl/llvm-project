// RUN: spm-opt -split-input-file -verify-diagnostics %s

// Mixing tensor and buffer-like (!spm.l2mem) operands on the same DPS op is
// rejected, exactly as mixing tensor and memref operands already is.
func.func @matmul_mixed_tensor_and_l2mem(%a: tensor<64x32xf32>, %b: !spm.l2mem<32x128xf32>,
                                           %c: !spm.l2mem<64x128xf32>) {
  // expected-error @below {{expected to have pure tensor or buffer semantics}}
  linalg.matmul ins(%a, %b : tensor<64x32xf32>, !spm.l2mem<32x128xf32>)
                outs(%c : !spm.l2mem<64x128xf32>)
  return
}
