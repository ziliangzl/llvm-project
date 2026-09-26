// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries test-analysis-only" | FileCheck %s

// CHECK-LABEL: func.func @repetitive_region_path_cache
func.func @repetitive_region_path_cache(%value: f32, %idx: index, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
    scf.if %cond {
      scf.if %cond {
        scf.if %cond {
          scf.if %cond {
            scf.if %cond {
              %empty = tensor.empty() : tensor<16xf32>
              // CHECK: linalg.fill
              // CHECK-SAME: __inplace_operands_attr__ = ["none", "true"]
              %filled = linalg.fill ins(%value : f32)
                  outs(%empty : tensor<16xf32>) -> tensor<16xf32>
              %read = tensor.extract %filled[%idx] : tensor<16xf32>
              // CHECK: tensor.insert
              // CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
              %written = tensor.insert %read into %filled[%idx]
                  : tensor<16xf32>
              scf.yield
            }
            scf.if %cond {
              %empty = tensor.empty() : tensor<16xf32>
              // CHECK: linalg.fill
              // CHECK-SAME: __inplace_operands_attr__ = ["none", "true"]
              %filled = linalg.fill ins(%value : f32)
                  outs(%empty : tensor<16xf32>) -> tensor<16xf32>
              %read = tensor.extract %filled[%idx] : tensor<16xf32>
              // CHECK: tensor.insert
              // CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
              %written = tensor.insert %read into %filled[%idx]
                  : tensor<16xf32>
              scf.yield
            }
            scf.yield
          }
          scf.yield
        }
        scf.yield
      }
      scf.yield
    }
  }
  return
}
