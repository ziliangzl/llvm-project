// RUN: spm-opt %s | spm-opt | FileCheck %s

// `linalg.matmul` operating directly on `!spm.l2mem` operands (in place of
// the usual memref/tensor), the same "buffer semantics, no results" shape it
// already supports for memref.
// CHECK-LABEL: func.func @matmul_on_l2mem
func.func @matmul_on_l2mem(%a: !spm.l2mem<64x32xf32>, %b: !spm.l2mem<32x128xf32>,
                             %c: !spm.l2mem<64x128xf32>) {
  // CHECK: linalg.matmul ins(%{{.*}}, %{{.*}} : !spm.l2mem<64x32xf32>, !spm.l2mem<32x128xf32>) outs(%{{.*}} : !spm.l2mem<64x128xf32>)
  linalg.matmul ins(%a, %b : !spm.l2mem<64x32xf32>, !spm.l2mem<32x128xf32>)
                outs(%c : !spm.l2mem<64x128xf32>)
  return
}
