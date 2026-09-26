// REQUIRES: asserts
// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries test-analysis-only" -mlir-pass-statistics 2>&1 | FileCheck %s

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
              %filled = linalg.fill ins(%value : f32)
                  outs(%empty : tensor<16xf32>) -> tensor<16xf32>
              %read = tensor.extract %filled[%idx] : tensor<16xf32>
              %written = tensor.insert %read into %filled[%idx]
                  : tensor<16xf32>
              scf.yield
            }
            scf.if %cond {
              %empty = tensor.empty() : tensor<16xf32>
              %filled = linalg.fill ins(%value : f32)
                  outs(%empty : tensor<16xf32>) -> tensor<16xf32>
              %read = tensor.extract %filled[%idx] : tensor<16xf32>
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

// CHECK: OneShotBufferize
// CHECK-DAG: (S) {{[1-9][0-9]*}} num-repetitive-region-query
// CHECK-DAG: (S) {{[1-9][0-9]*}} num-repetitive-region-cache-hit
// CHECK-DAG: (S) {{[1-9][0-9]*}} num-repetitive-region-path-cache-hit
// CHECK-DAG: (S) {{[1-9][0-9]*}} num-repetitive-region-ancestor-step
