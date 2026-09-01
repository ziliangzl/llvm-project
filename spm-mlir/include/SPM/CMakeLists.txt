add_mlir_dialect(SPMOps spm)
add_mlir_doc(SPMDialect SPMDialect SPM/ -gen-dialect-doc)
add_mlir_doc(SPMOps SPMOps SPM/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS SPMPasses.td)
mlir_tablegen(SPMPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRSPMPassesIncGen)
