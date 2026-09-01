add_mlir_dialect_library(MLIRSPM
        SPMTypes.cpp
        SPMDialect.cpp
        SPMOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/SPM

        DEPENDS
        MLIRSPMOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )

add_subdirectory(Transforms)
