// ============================================================================
// AUTO-GENERATED I3C CSR Traits Definitions
// Generated from PeakRDL package files to abstract configuration types.
// ============================================================================

`ifndef I3C_CSR_TRAITS_SVH
`define I3C_CSR_TRAITS_SVH

// ---------------------------------------------------------
// Class: controller_and_target_csr_t
// ---------------------------------------------------------
class controller_and_target_csr_t;
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__out_t                          hwif_out_t;
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__in_t                           hwif_in_t;
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__I3CBase__out_t                 base_out_t;
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__I3CBase__in_t                  base_in_t;
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__PIOControl__out_t              pio_out_t;
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__PIOControl__in_t               pio_in_t;
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__I3C_EC__out_t                  ec_out_t;
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__I3C_EC__in_t                   ec_in_t;
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__I3C_EC__SecFwRecoveryIf__out_t secfwrecoveryif_out_t;
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__I3C_EC__SecFwRecoveryIf__in_t  secfwrecoveryif_in_t;
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__I3C_EC__StdbyCtrlMode__out_t   stdby_out_t;
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__I3C_EC__StdbyCtrlMode__in_t    stdby_in_t;
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__I3C_EC__SoCMgmtIf__out_t       socmgmt_out_t;
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__I3C_EC__SoCMgmtIf__in_t        socmgmt_in_t;
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__I3C_EC__CtrlCfg__out_t         ctrlcfg_out_t;
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__I3C_EC__CtrlCfg__in_t          ctrlcfg_in_t;
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__I3C_EC__TTI__in_t              tti_in_t;
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__I3C_EC__TTI__out_t             tti_out_t;
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__DAT__out_t                     dat_out_t;
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__DAT__in_t                      dat_in_t;
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__DCT__out_t                     dct_out_t;
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__DCT__in_t                      dct_in_t;
endclass

// ---------------------------------------------------------
// Class: target_csr_t
// ---------------------------------------------------------
class target_csr_t;
  typedef target_I3CCSR_pkg::target_I3CCSR__out_t                                         hwif_out_t;
  typedef target_I3CCSR_pkg::target_I3CCSR__in_t                                          hwif_in_t;
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__I3CBase__out_t base_out_t; // DUMMY (Borrowed from combo for parser)
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__I3CBase__in_t base_in_t; // DUMMY (Borrowed from combo for parser)
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__PIOControl__out_t pio_out_t; // DUMMY (Borrowed from combo for parser)
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__PIOControl__in_t pio_in_t; // DUMMY (Borrowed from combo for parser)
  typedef target_I3CCSR_pkg::target_I3CCSR__I3C_EC__out_t                                 ec_out_t;
  typedef target_I3CCSR_pkg::target_I3CCSR__I3C_EC__in_t                                  ec_in_t;
  typedef target_I3CCSR_pkg::target_I3CCSR__I3C_EC__SecFwRecoveryIf__out_t                secfwrecoveryif_out_t;
  typedef target_I3CCSR_pkg::target_I3CCSR__I3C_EC__SecFwRecoveryIf__in_t                 secfwrecoveryif_in_t;
  typedef target_I3CCSR_pkg::target_I3CCSR__I3C_EC__StdbyCtrlMode__out_t                  stdby_out_t;
  typedef target_I3CCSR_pkg::target_I3CCSR__I3C_EC__StdbyCtrlMode__in_t                   stdby_in_t;
  typedef target_I3CCSR_pkg::target_I3CCSR__I3C_EC__SoCMgmtIf__out_t                      socmgmt_out_t;
  typedef target_I3CCSR_pkg::target_I3CCSR__I3C_EC__SoCMgmtIf__in_t                       socmgmt_in_t;
  typedef target_I3CCSR_pkg::target_I3CCSR__I3C_EC__CtrlCfg__out_t                        ctrlcfg_out_t;
  typedef target_I3CCSR_pkg::target_I3CCSR__I3C_EC__CtrlCfg__in_t                         ctrlcfg_in_t;
  typedef target_I3CCSR_pkg::target_I3CCSR__I3C_EC__TTI__in_t                             tti_in_t;
  typedef target_I3CCSR_pkg::target_I3CCSR__I3C_EC__TTI__out_t                            tti_out_t;
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__DAT__out_t dat_out_t; // DUMMY (Borrowed from combo for parser)
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__DAT__in_t dat_in_t; // DUMMY (Borrowed from combo for parser)
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__DCT__out_t dct_out_t; // DUMMY (Borrowed from combo for parser)
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__DCT__in_t dct_in_t; // DUMMY (Borrowed from combo for parser)
endclass

// ---------------------------------------------------------
// Class: controller_csr_t
// ---------------------------------------------------------
class controller_csr_t;
  typedef controller_I3CCSR_pkg::controller_I3CCSR__out_t                                     hwif_out_t;
  typedef controller_I3CCSR_pkg::controller_I3CCSR__in_t                                      hwif_in_t;
  typedef controller_I3CCSR_pkg::controller_I3CCSR__I3CBase__out_t                            base_out_t;
  typedef controller_I3CCSR_pkg::controller_I3CCSR__I3CBase__in_t                             base_in_t;
  typedef controller_I3CCSR_pkg::controller_I3CCSR__PIOControl__out_t                         pio_out_t;
  typedef controller_I3CCSR_pkg::controller_I3CCSR__PIOControl__in_t                          pio_in_t;
  typedef controller_I3CCSR_pkg::controller_I3CCSR__I3C_EC__out_t                             ec_out_t;
  typedef controller_I3CCSR_pkg::controller_I3CCSR__I3C_EC__in_t                              ec_in_t;
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__I3C_EC__SecFwRecoveryIf__out_t secfwrecoveryif_out_t; // DUMMY (Borrowed from combo for parser)
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__I3C_EC__SecFwRecoveryIf__in_t secfwrecoveryif_in_t; // DUMMY (Borrowed from combo for parser)
  typedef controller_I3CCSR_pkg::controller_I3CCSR__I3C_EC__StdbyCtrlMode__out_t              stdby_out_t;
  typedef controller_I3CCSR_pkg::controller_I3CCSR__I3C_EC__StdbyCtrlMode__in_t               stdby_in_t;
  typedef controller_I3CCSR_pkg::controller_I3CCSR__I3C_EC__SoCMgmtIf__out_t                  socmgmt_out_t;
  typedef controller_I3CCSR_pkg::controller_I3CCSR__I3C_EC__SoCMgmtIf__in_t                   socmgmt_in_t;
  typedef controller_I3CCSR_pkg::controller_I3CCSR__I3C_EC__CtrlCfg__out_t                    ctrlcfg_out_t;
  typedef controller_I3CCSR_pkg::controller_I3CCSR__I3C_EC__CtrlCfg__in_t                     ctrlcfg_in_t;
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__I3C_EC__TTI__in_t tti_in_t; // DUMMY (Borrowed from combo for parser)
  typedef controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR__I3C_EC__TTI__out_t tti_out_t; // DUMMY (Borrowed from combo for parser)
  typedef controller_I3CCSR_pkg::controller_I3CCSR__DAT__out_t                                dat_out_t;
  typedef controller_I3CCSR_pkg::controller_I3CCSR__DAT__in_t                                 dat_in_t;
  typedef controller_I3CCSR_pkg::controller_I3CCSR__DCT__out_t                                dct_out_t;
  typedef controller_I3CCSR_pkg::controller_I3CCSR__DCT__in_t                                 dct_in_t;
endclass

`endif // I3C_CSR_TRAITS_SVH
