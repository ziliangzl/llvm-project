// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries" -split-input-file | FileCheck %s
// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries test-analysis-only" -split-input-file | FileCheck %s -check-prefix=CHECK-ANALYSIS

// CHECK-LABEL: func @mask(
//  CHECK-SAME:     %[[t0:.*]]: memref<?xf32, strided<[?], offset: ?>>
func.func @mask(%t0: tensor<?xf32>, %val: vector<16xf32>, %idx: index, %m0: vector<16xi1>) -> tensor<?xf32> {
  // CHECK-NOT: alloc
  // CHECK-NOT: copy
  //     CHECK: vector.mask %{{.*}} { vector.transfer_write %{{.*}}, %[[t0]][%{{.*}}] : vector<16xf32>, memref<?xf32, strided<[?], offset: ?>> } : vector<16xi1>
  %0 = vector.mask %m0 { vector.transfer_write %val, %t0[%idx] : vector<16xf32>, tensor<?xf32> } : vector<16xi1> -> tensor<?xf32>
  //     CHECK: return %[[t0]]
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @mask_scalable(
//  CHECK-SAME:     %[[t0:.*]]: memref<?xf32, strided<[?], offset: ?>>
func.func @mask_scalable(%t0: tensor<?xf32>, %val: vector<[16]xf32>, %idx: index, %m0: vector<[16]xi1>) -> tensor<?xf32> {
  // CHECK-NOT: alloc
  // CHECK-NOT: copy
  //     CHECK: vector.mask %{{.*}} { vector.transfer_write %{{.*}}, %[[t0]][%{{.*}}] : vector<[16]xf32>, memref<?xf32, strided<[?], offset: ?>> } : vector<[16]xi1>
  %0 = vector.mask %m0 { vector.transfer_write %val, %t0[%idx] : vector<[16]xf32>, tensor<?xf32> } : vector<[16]xi1> -> tensor<?xf32>
  //     CHECK: return %[[t0]]
  return %0 : tensor<?xf32>
}

// -----

// CHECK-ANALYSIS-LABEL: func @non_reading_xfer_write(
//  CHECK-ANALYSIS-SAME: tensor<5x10xf32> {bufferization.access = "write"}
func.func @non_reading_xfer_write(%t: tensor<5x10xf32>, %v: vector<6x11xf32>) -> tensor<5x10xf32> {
  %c0 = arith.constant 0 : index
  %1 = vector.transfer_write %v, %t[%c0, %c0] : vector<6x11xf32>, tensor<5x10xf32>
  return %1 : tensor<5x10xf32>
}
// -----

// CHECK-ANALYSIS-LABEL: func @reading_xfer_write(
//  CHECK-ANALYSIS-SAME: tensor<5x10xf32> {bufferization.access = "read-write"}
func.func @reading_xfer_write(%t: tensor<5x10xf32>, %v: vector<4x11xf32>) -> tensor<5x10xf32> {
  %c0 = arith.constant 0 : index
  %1 = vector.transfer_write %v, %t[%c0, %c0] : vector<4x11xf32>, tensor<5x10xf32>
  return %1 : tensor<5x10xf32>
}

// -----

// A write to [0, 4) does not conflict with a read from [4, 8), so the write
// can bufferize in-place.

// CHECK-LABEL: func @disjoint_transfer_read_write(
//  CHECK-SAME:     %[[T:.*]]: memref<8xf32, strided<[?], offset: ?>>
//   CHECK-NOT:   memref.alloc
//   CHECK-NOT:   memref.copy
//       CHECK:   vector.transfer_write %{{.*}}, %[[T]][%{{.*}}]
//       CHECK:   vector.transfer_read %[[T]][%{{.*}}]
//       CHECK:   return %[[T]],

// CHECK-ANALYSIS-LABEL: func @disjoint_transfer_read_write(
// CHECK-ANALYSIS: vector.transfer_write
// CHECK-ANALYSIS-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
func.func @disjoint_transfer_read_write(
    %t: tensor<8xf32> {bufferization.writable = true},
    %v: vector<4xf32>) -> (tensor<8xf32>, vector<4xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %pad = arith.constant 0.0 : f32
  %written = vector.transfer_write %v, %t[%c0]
      : vector<4xf32>, tensor<8xf32>
  %read = vector.transfer_read %t[%c4], %pad
      : tensor<8xf32>, vector<4xf32>
  return %written, %read : tensor<8xf32>, vector<4xf32>
}

// -----

// A write to [0, 4) conflicts with a read from [2, 6), so the write must
// bufferize out-of-place.

// CHECK-LABEL: func @overlapping_transfer_read_write(
//       CHECK:   %[[ALLOC:.*]] = memref.alloc
//       CHECK:   memref.copy %{{.*}}, %[[ALLOC]]

// CHECK-ANALYSIS-LABEL: func @overlapping_transfer_read_write(
// CHECK-ANALYSIS: vector.transfer_write
// CHECK-ANALYSIS-SAME: __inplace_operands_attr__ = ["none", "false", "none"]
func.func @overlapping_transfer_read_write(
    %t: tensor<8xf32> {bufferization.writable = true},
    %v: vector<4xf32>) -> (tensor<8xf32>, vector<4xf32>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %pad = arith.constant 0.0 : f32
  %written = vector.transfer_write %v, %t[%c0]
      : vector<4xf32>, tensor<8xf32>
  %read = vector.transfer_read %t[%c2], %pad
      : tensor<8xf32>, vector<4xf32>
  return %written, %read : tensor<8xf32>, vector<4xf32>
}

// -----

// If the subsets cannot be proven disjoint, the write must bufferize
// out-of-place.

// CHECK-LABEL: func @unknown_transfer_read_write(
//       CHECK:   %[[ALLOC:.*]] = memref.alloc
//       CHECK:   memref.copy %{{.*}}, %[[ALLOC]]

// CHECK-ANALYSIS-LABEL: func @unknown_transfer_read_write(
// CHECK-ANALYSIS: vector.transfer_write
// CHECK-ANALYSIS-SAME: __inplace_operands_attr__ = ["none", "false", "none"]
func.func @unknown_transfer_read_write(
    %t: tensor<8xf32> {bufferization.writable = true},
    %v: vector<4xf32>, %write_idx: index,
    %read_idx: index) -> (tensor<8xf32>, vector<4xf32>) {
  %pad = arith.constant 0.0 : f32
  %written = vector.transfer_write %v, %t[%write_idx]
      : vector<4xf32>, tensor<8xf32>
  %read = vector.transfer_read %t[%read_idx], %pad
      : tensor<8xf32>, vector<4xf32>
  return %written, %read : tensor<8xf32>, vector<4xf32>
}

// -----

// The write is enclosed by the lhs extraction. That extraction is disjoint
// from the rhs extraction, so the transfer_write can bufferize in-place.

// CHECK-LABEL: func @disjoint_transfer_write_extraction_provenance(
//   CHECK-NOT:   memref.alloc
//       CHECK:   vector.transfer_write
//   CHECK-NOT:   memref.alloc
//       CHECK:   return

// CHECK-ANALYSIS-LABEL: func @disjoint_transfer_write_extraction_provenance(
// CHECK-ANALYSIS: tensor.extract_slice
// CHECK-ANALYSIS-SAME: __inplace_operands_attr__ = ["true"]
// CHECK-ANALYSIS: vector.transfer_write
// CHECK-ANALYSIS-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
func.func @disjoint_transfer_write_extraction_provenance(
    %t: tensor<8xf32> {bufferization.writable = true},
    %v: vector<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %c0 = arith.constant 0 : index
  %lhs = tensor.extract_slice %t[0][4][1]
      : tensor<8xf32> to tensor<4xf32>
  %rhs = tensor.extract_slice %t[4][4][1]
      : tensor<8xf32> to tensor<4xf32>
  %written = vector.transfer_write %v, %lhs[%c0]
      : vector<4xf32>, tensor<4xf32>
  return %written, %rhs : tensor<4xf32>, tensor<4xf32>
}

// -----

// The enclosing write extraction overlaps the read extraction, so the write
// path must be separated from the original buffer.

// CHECK-LABEL: func @overlapping_transfer_write_extraction_provenance(
//       CHECK:   %[[ALLOC:.*]] = memref.alloc

// CHECK-ANALYSIS-LABEL: func @overlapping_transfer_write_extraction_provenance(
// CHECK-ANALYSIS: tensor.extract_slice
// CHECK-ANALYSIS-SAME: __inplace_operands_attr__ = ["false"]
// CHECK-ANALYSIS: vector.transfer_write
// CHECK-ANALYSIS-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
func.func @overlapping_transfer_write_extraction_provenance(
    %t: tensor<8xf32> {bufferization.writable = true},
    %v: vector<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %c0 = arith.constant 0 : index
  %lhs = tensor.extract_slice %t[0][4][1]
      : tensor<8xf32> to tensor<4xf32>
  %rhs = tensor.extract_slice %t[2][4][1]
      : tensor<8xf32> to tensor<4xf32>
  %written = vector.transfer_write %v, %lhs[%c0]
      : vector<4xf32>, tensor<4xf32>
  return %written, %rhs : tensor<4xf32>, tensor<4xf32>
}

// -----

// Every possible writer origin is disjoint from the read subset.

// CHECK-LABEL: func @disjoint_multi_write_origins(
//   CHECK-NOT:   memref.alloc
//       CHECK:   arith.select
//   CHECK-NOT:   memref.alloc
//       CHECK:   vector.transfer_write
//   CHECK-NOT:   memref.alloc
//       CHECK:   return

// CHECK-ANALYSIS-LABEL: func @disjoint_multi_write_origins(
// CHECK-ANALYSIS: tensor.extract_slice
// CHECK-ANALYSIS-SAME: __inplace_operands_attr__ = ["true"]
// CHECK-ANALYSIS: tensor.extract_slice
// CHECK-ANALYSIS-SAME: __inplace_operands_attr__ = ["true"]
// CHECK-ANALYSIS: arith.select
// CHECK-ANALYSIS-SAME: __inplace_operands_attr__ = ["none", "true", "true"]
// CHECK-ANALYSIS: vector.transfer_write
// CHECK-ANALYSIS-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
func.func @disjoint_multi_write_origins(
    %t: tensor<12xf32> {bufferization.writable = true},
    %v: vector<2xf32>, %cond: i1) -> (tensor<2xf32>, tensor<4xf32>) {
  %c0 = arith.constant 0 : index
  %write_a = tensor.extract_slice %t[0][2][1]
      : tensor<12xf32> to tensor<2xf32>
  %write_b = tensor.extract_slice %t[2][2][1]
      : tensor<12xf32> to tensor<2xf32>
  %write = arith.select %cond, %write_a, %write_b : tensor<2xf32>
  %read = tensor.extract_slice %t[8][4][1]
      : tensor<12xf32> to tensor<4xf32>
  %written = vector.transfer_write %v, %write[%c0]
      : vector<2xf32>, tensor<2xf32>
  return %written, %read : tensor<2xf32>, tensor<4xf32>
}

// -----

// One possible writer origin overlaps the read subset, so the write path must
// be separated from the original buffer.

// CHECK-LABEL: func @overlapping_multi_write_origins(
//       CHECK:   %[[ALLOC:.*]] = memref.alloc

// CHECK-ANALYSIS-LABEL: func @overlapping_multi_write_origins(
// CHECK-ANALYSIS: tensor.extract_slice
// CHECK-ANALYSIS-SAME: __inplace_operands_attr__ = ["false"]
// CHECK-ANALYSIS: tensor.extract_slice
// CHECK-ANALYSIS-SAME: __inplace_operands_attr__ = ["false"]
// CHECK-ANALYSIS: arith.select
// CHECK-ANALYSIS-SAME: __inplace_operands_attr__ = ["none", "true", "true"]
// CHECK-ANALYSIS: vector.transfer_write
// CHECK-ANALYSIS-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
func.func @overlapping_multi_write_origins(
    %t: tensor<12xf32> {bufferization.writable = true},
    %v: vector<2xf32>, %cond: i1) -> (tensor<2xf32>, tensor<4xf32>) {
  %c0 = arith.constant 0 : index
  %write_a = tensor.extract_slice %t[0][2][1]
      : tensor<12xf32> to tensor<2xf32>
  %write_b = tensor.extract_slice %t[8][2][1]
      : tensor<12xf32> to tensor<2xf32>
  %write = arith.select %cond, %write_a, %write_b : tensor<2xf32>
  %read = tensor.extract_slice %t[8][4][1]
      : tensor<12xf32> to tensor<4xf32>
  %written = vector.transfer_write %v, %write[%c0]
      : vector<2xf32>, tensor<2xf32>
  return %written, %read : tensor<2xf32>, tensor<4xf32>
}
