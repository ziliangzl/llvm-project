// RUN: spm-opt -split-input-file -verify-diagnostics %s

func.func @store_shape_mismatch(%buf: !spm.l1mem<64x64xf32>, %val: tensor<32x32xf32>) {
  // expected-error @below {{buffer shape does not match tensor shape}}
  spm.l1_store %val, %buf : tensor<32x32xf32> -> !spm.l1mem<64x64xf32>
  return
}

// -----

func.func @store_elemtype_mismatch(%buf: !spm.l1mem<64x64xf32>, %val: tensor<64x64xi32>) {
  // expected-error @below {{buffer element type 'f32' does not match tensor element type 'i32'}}
  spm.l1_store %val, %buf : tensor<64x64xi32> -> !spm.l1mem<64x64xf32>
  return
}

// -----

func.func @store_into_immutable(%buf: !spm.l1mem<64x64xf32, immutable>, %val: tensor<64x64xf32>) {
  // expected-error @below {{cannot store into an immutable buffer}}
  spm.l1_store %val, %buf : tensor<64x64xf32> -> !spm.l1mem<64x64xf32, immutable>
  return
}

// -----

func.func @l2_load_shape_mismatch(%buf: !spm.l2mem<8x8xf16>) -> tensor<4x4xf16> {
  // expected-error @below {{buffer shape does not match tensor shape}}
  %0 = spm.l2_load %buf : !spm.l2mem<8x8xf16> -> tensor<4x4xf16>
  return %0 : tensor<4x4xf16>
}

// -----

// expected-error @below {{buffer dimensions must be positive}}
func.func @zero_dim(%buf: !spm.l1mem<0x4xf32>) {
  return
}

// -----

// expected-error @below {{alloc shape must have at least as many dimensions as the buffer shape}}
func.func @alloc_shape_too_small(%buf: !spm.l1mem<2x3x4xf32, alloc<3x4>>) {
  return
}
