// RUN: spm-opt %s | spm-opt | FileCheck %s

// CHECK-LABEL: func.func @l1_roundtrip
// CHECK-SAME: (%[[BUF:.*]]: !spm.l1mem<64x64xf32>)
func.func @l1_roundtrip(%buf: !spm.l1mem<64x64xf32>) -> tensor<64x64xf32> {
  // CHECK: %[[V:.*]] = spm.l1_load %[[BUF]] : <64x64xf32> -> tensor<64x64xf32>
  %0 = spm.l1_load %buf : !spm.l1mem<64x64xf32> -> tensor<64x64xf32>
  // CHECK: spm.l1_store %[[V]], %[[BUF]] : tensor<64x64xf32> -> <64x64xf32>
  spm.l1_store %0, %buf : tensor<64x64xf32> -> !spm.l1mem<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @l2_roundtrip
// CHECK-SAME: (%[[BUF:.*]]: !spm.l2mem<128x32xi16>)
func.func @l2_roundtrip(%buf: !spm.l2mem<128x32xi16>) -> tensor<128x32xi16> {
  // CHECK: %[[V:.*]] = spm.l2_load %[[BUF]] : <128x32xi16> -> tensor<128x32xi16>
  %0 = spm.l2_load %buf : !spm.l2mem<128x32xi16> -> tensor<128x32xi16>
  // CHECK: spm.l2_store %[[V]], %[[BUF]] : tensor<128x32xi16> -> <128x32xi16>
  spm.l2_store %0, %buf : tensor<128x32xi16> -> !spm.l2mem<128x32xi16>
  return %0 : tensor<128x32xi16>
}

// CHECK-LABEL: func.func @l1_immutable
// CHECK-SAME: (%[[BUF:.*]]: !spm.l1mem<16xf32, immutable>)
func.func @l1_immutable(%buf: !spm.l1mem<16xf32, immutable>) -> tensor<16xf32> {
  // CHECK: spm.l1_load %[[BUF]] : <16xf32, immutable> -> tensor<16xf32>
  %0 = spm.l1_load %buf : !spm.l1mem<16xf32, immutable> -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// A slice of a multi-buffered L1 allocation: `shape` is one buffer's worth,
// `allocShape` records the full ring of 3.
// CHECK-LABEL: func.func @l1_multibuffer_slice
// CHECK-SAME: (%[[BUF:.*]]: !spm.l1mem<64x64xf32, alloc<3x64x64>>)
func.func @l1_multibuffer_slice(%buf: !spm.l1mem<64x64xf32, alloc<3x64x64>>) -> tensor<64x64xf32> {
  // CHECK: spm.l1_load %[[BUF]] : <64x64xf32, alloc<3x64x64>> -> tensor<64x64xf32>
  %0 = spm.l1_load %buf : !spm.l1mem<64x64xf32, alloc<3x64x64>> -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}
