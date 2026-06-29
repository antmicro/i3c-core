// SPDX-License-Identifier: Apache-2.0
// This wrapper module provides compliance to cocotb-AHB
// AHB signal naming convention
module ahb_if_wrapper
  import i3c_pkg::*;
#(
    parameter type csr_cfg_t = target_csr_t,
    parameter bit ControllerEn = 0,  // enables host controller configuration
    parameter bit TargetEn = 1,  // enables target configuration
    parameter int unsigned CsrAddrWidth = (ControllerEn && TargetEn) ? controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR_MIN_ADDR_WIDTH :
                               (ControllerEn)             ? controller_I3CCSR_pkg::controller_I3CCSR_MIN_ADDR_WIDTH :
                                                            target_I3CCSR_pkg::target_I3CCSR_MIN_ADDR_WIDTH,
    parameter int unsigned CsrDataWidth = (ControllerEn && TargetEn) ? controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR_DATA_WIDTH :
                               (ControllerEn)             ? controller_I3CCSR_pkg::controller_I3CCSR_DATA_WIDTH :
                                                            target_I3CCSR_pkg::target_I3CCSR_DATA_WIDTH,

    parameter int unsigned AhbDataWidth = 64,
    parameter int unsigned AhbAddrWidth = 32
) (
    // AHB-Lite interface
    input  logic                      hclk,
    input  logic                      hreset_n,
    input  logic [  AhbAddrWidth-1:0] haddr,
    input  logic [               2:0] hburst,
    input  logic [               3:0] hprot,
    input  logic [               2:0] hsize,
    input  logic [               1:0] htrans,
    input  logic [  AhbDataWidth-1:0] hwdata,
    input  logic [AhbDataWidth/8-1:0] hwstrb,
    input  logic                      hwrite,
    output logic [  AhbDataWidth-1:0] hrdata,
    output logic                      hreadyout,
    output logic                      hresp,
    input  logic                      hsel,
    input  logic                      hready
);
  // I3C SW CSR access interface
  logic                    s_cpuif_req;
  logic                    s_cpuif_req_is_wr;
  logic [CsrAddrWidth-1:0] s_cpuif_addr;
  logic [CsrDataWidth-1:0] s_cpuif_wr_data;
  logic [CsrDataWidth-1:0] s_cpuif_wr_biten;
  logic                    s_cpuif_req_stall_wr;
  logic                    s_cpuif_req_stall_rd;
  logic                    s_cpuif_rd_ack;
  logic                    s_cpuif_rd_err;
  logic [CsrDataWidth-1:0] s_cpuif_rd_data;
  logic                    s_cpuif_wr_ack;
  logic                    s_cpuif_wr_err;

  ahb_if #(
      .AhbDataWidth(AhbDataWidth),
      .AhbAddrWidth(AhbAddrWidth)
  ) i3c_ahb_if (
      .hclk_i(hclk),
      .hreset_n_i(hreset_n),
      .haddr_i(haddr),
      .hburst_i(hburst),
      .hprot_i(hprot),
      .hsize_i(hsize),
      .htrans_i(htrans),
      .hwdata_i(hwdata),
      .hwstrb_i(hwstrb),
      .hwrite_i(hwrite),
      .hrdata_o(hrdata),
      .hreadyout_o(hreadyout),
      .hresp_o(hresp),
      .hsel_i(hsel),
      .hready_i(hready),
      .s_cpuif_req(s_cpuif_req),
      .s_cpuif_req_is_wr(s_cpuif_req_is_wr),
      .s_cpuif_addr(s_cpuif_addr),
      .s_cpuif_wr_data(s_cpuif_wr_data),
      .s_cpuif_wr_biten(s_cpuif_wr_biten),
      .s_cpuif_req_stall_wr(s_cpuif_req_stall_wr),
      .s_cpuif_req_stall_rd(s_cpuif_req_stall_rd),
      .s_cpuif_rd_ack(s_cpuif_rd_ack),
      .s_cpuif_rd_err(s_cpuif_rd_err),
      .s_cpuif_rd_data(s_cpuif_rd_data),
      .s_cpuif_wr_ack(s_cpuif_wr_ack),
      .s_cpuif_wr_err(s_cpuif_wr_err)
  );

  csr_cfg_t::hwif_in_t  hwif_in;
  csr_cfg_t::hwif_out_t hwif_out;

  assign hwif_in.rst_ni = hreset_n;

  // Connect to I3C CSRs to test SW access
  // FUTUREFIX: #99552 add controller_csr and controller_and_target_csr
  target_I3CCSR i3c_csr (
      .clk(hclk),
      .rst(~hreset_n),

      .s_cpuif_req(s_cpuif_req),
      .s_cpuif_req_is_wr(s_cpuif_req_is_wr),
      .s_cpuif_addr(s_cpuif_addr),
      .s_cpuif_wr_data(s_cpuif_wr_data),
      .s_cpuif_wr_biten(s_cpuif_wr_biten),  // Write strobes not handled by AHB-Lite interface
      .s_cpuif_req_stall_wr(s_cpuif_req_stall_wr),
      .s_cpuif_req_stall_rd(s_cpuif_req_stall_rd),
      .s_cpuif_rd_ack(s_cpuif_rd_ack),  // Ignored by AHB component
      .s_cpuif_rd_err(s_cpuif_rd_err),
      .s_cpuif_rd_data(s_cpuif_rd_data),
      .s_cpuif_wr_ack(s_cpuif_wr_ack),  // Ignored by AHB component
      .s_cpuif_wr_err(s_cpuif_wr_err),

      .hwif_in (hwif_in),
      .hwif_out(hwif_out)
  );

  always_comb begin : missing_csr_we_inits
    hwif_in.I3C_EC.StdbyCtrlMode.STBY_CR_CONTROL.HANDOFF_DEEP_SLEEP.we = 0;
    hwif_in.I3C_EC.StdbyCtrlMode.STBY_CR_CONTROL.TARGET_XACT_ENABLE.we = 0;
    hwif_in.I3C_EC.StdbyCtrlMode.STBY_CR_CONTROL.DAA_SETAASA_ENABLE.we = 0;
    hwif_in.I3C_EC.StdbyCtrlMode.STBY_CR_CONTROL.DAA_SETDASA_ENABLE.we = 0;
    hwif_in.I3C_EC.StdbyCtrlMode.STBY_CR_CONTROL.DAA_ENTDAA_ENABLE.we = 0;
    hwif_in.I3C_EC.TTI.RESET_CONTROL.SOFT_RST.we = 0;
    hwif_in.I3C_EC.TTI.RESET_CONTROL.TX_DESC_RST.we = 0;
    hwif_in.I3C_EC.TTI.RESET_CONTROL.RX_DESC_RST.we = 0;
    hwif_in.I3C_EC.TTI.RESET_CONTROL.TX_DATA_RST.we = 0;
    hwif_in.I3C_EC.TTI.RESET_CONTROL.RX_DATA_RST.we = 0;
    hwif_in.I3C_EC.TTI.RESET_CONTROL.IBI_QUEUE_RST.we = 0;
    hwif_in.I3C_EC.TTI.QUEUE_THLD_CTRL.TX_DESC_THLD.we = 0;
    hwif_in.I3C_EC.TTI.QUEUE_THLD_CTRL.RX_DESC_THLD.we = 0;
    hwif_in.I3C_EC.TTI.QUEUE_THLD_CTRL.IBI_THLD.we = 0;
    hwif_in.I3C_EC.CtrlCfg.CONTROLLER_CONFIG.OPERATION_MODE.we = 0;
  end : missing_csr_we_inits

  always_comb begin : other_uninit_signals
    hwif_in.I3C_EC.StdbyCtrlMode.STBY_CR_CONTROL.HANDOFF_DEEP_SLEEP.hwclr = 0;

    // Unhandled wr/rd_ack (drivers are not included in this wrapper)
    hwif_in.I3C_EC.TTI.RX_DESC_QUEUE_PORT.rd_ack = 0;
    hwif_in.I3C_EC.TTI.RX_DATA_PORT.rd_ack = 0;
    hwif_in.I3C_EC.TTI.TX_DESC_QUEUE_PORT.wr_ack = 0;
    hwif_in.I3C_EC.TTI.TX_DATA_PORT.wr_ack = 0;
    hwif_in.I3C_EC.TTI.IBI_PORT.wr_ack = 0;
    hwif_in.I3C_EC.SecFwRecoveryIf.INDIRECT_FIFO_DATA.rd_ack = '0;

  end : other_uninit_signals
endmodule
