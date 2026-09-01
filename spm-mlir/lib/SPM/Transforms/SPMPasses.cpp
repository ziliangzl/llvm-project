//===- SPMPasses.cpp - SPM dialect passes ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPM/SPMPasses.h"

#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::spm {
#define GEN_PASS_DEF_SPMLOWERLINALGMATMULTOL2
#include "SPM/SPMPasses.h.inc"

namespace {

/// Lowers a `linalg.matmul` whose `ins`/`outs` are all `!spm.l2mem` into:
///   %a = spm.l2_load %A ...
///   %b = spm.l2_load %B ...
///   %c_init = spm.l2_load %C ...
///   %r = linalg.matmul ins(%a, %b) outs(%c_init) -> tensor<...>
///   spm.l2_store %r, %C ...
struct LowerL2MatmulPattern : OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                 PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 0 ||
        !llvm::all_of(op->getOperandTypes(),
                       [](Type t) { return isa<L2MemType>(t); }))
      return rewriter.notifyMatchFailure(
          op, "expected a buffer-semantics matmul with all !spm.l2mem "
              "operands");

    Location loc = op.getLoc();
    auto loadOperand = [&](Value buf) -> Value {
      auto descTy = cast<L2MemType>(buf.getType());
      auto tensorTy =
          RankedTensorType::get(descTy.getShape(), descTy.getElementType());
      return L2LoadOp::create(rewriter, loc, tensorTy, buf);
    };

    SmallVector<Value> loadedInputs;
    for (Value input : op.getDpsInputs())
      loadedInputs.push_back(loadOperand(input));

    Value dstBuf = op.getDpsInits()[0];
    Value loadedInit = loadOperand(dstBuf);

    auto newMatmul = linalg::MatmulOp::create(
        rewriter, loc, loadedInputs, ValueRange{loadedInit});

    L2StoreOp::create(rewriter, loc, newMatmul.getResult(0), dstBuf);
    rewriter.eraseOp(op);
    return success();
  }
};

class SPMLowerLinalgMatmulToL2
    : public impl::SPMLowerLinalgMatmulToL2Base<SPMLowerLinalgMatmulToL2> {
public:
  using impl::SPMLowerLinalgMatmulToL2Base<
      SPMLowerLinalgMatmulToL2>::SPMLowerLinalgMatmulToL2Base;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    populateLowerLinalgMatmulToL2Patterns(patterns);
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};

} // namespace

void populateLowerLinalgMatmulToL2Patterns(RewritePatternSet &patterns) {
  patterns.add<LowerL2MatmulPattern>(patterns.getContext());
}

} // namespace mlir::spm
