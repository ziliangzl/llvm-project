configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(SPM_TEST_DEPENDS
        FileCheck count not
        spm-opt
        )

add_lit_testsuite(check-spm "Running the spm-mlir regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${SPM_TEST_DEPENDS}
        )
set_target_properties(check-spm PROPERTIES FOLDER "Tests")

add_lit_testsuites(SPM ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${SPM_TEST_DEPENDS})
