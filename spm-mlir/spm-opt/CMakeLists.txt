set(LIBS
    MLIRArithDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIROptLib
    MLIRTensorDialect
    MLIRSPM
    MLIRSPMTransforms
  )
add_llvm_executable(spm-opt spm-opt.cpp)

llvm_update_compile_flags(spm-opt)
target_link_libraries(spm-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(spm-opt)
