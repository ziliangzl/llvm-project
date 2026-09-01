add_mlir_dialect_library(MLIRSPMTransforms
        SPMPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/SPM

        DEPENDS
        MLIRSPMPassesIncGen

        LINK_LIBS PUBLIC
        MLIRSPM
        MLIRIR
        MLIRPass
        MLIRRewrite
        MLIRTransforms
        MLIRFuncDialect
        MLIRLinalgDialect
        MLIRTensorDialect
        )
