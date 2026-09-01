//===- spm-opt.cpp --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "SPM/SPMDialect.h"
#include "SPM/SPMPasses.h"

int main(int argc, char **argv) {
  mlir::spm::registerPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::spm::SPMDialect, mlir::arith::ArithDialect,
                   mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                   mlir::tensor::TensorDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "SPM optimizer driver\n", registry));
}
