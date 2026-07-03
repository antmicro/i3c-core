// SPDX-License-Identifier: Apache-2.0

`ifndef I3C_CSR_CFG_EXTRACTOR_DEF
`define I3C_CSR_CFG_EXTRACTOR_DEF
class csr_cfg_extractor #(
    type T
);
  typedef T::hwif_out_t hwif_out_t;
  typedef T::hwif_in_t hwif_in_t;
  typedef T::base_out_t base_out_t;
  typedef T::base_in_t base_in_t;
  typedef T::pio_out_t pio_out_t;
  typedef T::pio_in_t pio_in_t;
  typedef T::ec_out_t ec_out_t;
  typedef T::ec_in_t ec_in_t;
  typedef T::secfwrecoveryif_out_t secfwrecoveryif_out_t;
  typedef T::secfwrecoveryif_in_t secfwrecoveryif_in_t;
  typedef T::stdby_out_t stdby_out_t;
  typedef T::stdby_in_t stdby_in_t;
  typedef T::socmgmt_out_t socmgmt_out_t;
  typedef T::socmgmt_in_t socmgmt_in_t;
  typedef T::ctrlcfg_out_t ctrlcfg_out_t;
  typedef T::ctrlcfg_in_t ctrlcfg_in_t;
  typedef T::tti_in_t tti_in_t;
  typedef T::tti_out_t tti_out_t;
  typedef T::dat_out_t dat_out_t;
  typedef T::dat_in_t dat_in_t;
  typedef T::dct_out_t dct_out_t;
  typedef T::dct_in_t dct_in_t;
endclass
`endif  // I3C_CSR_CFG_EXTRACTOR_DEF
