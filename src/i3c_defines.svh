// SPDX-License-Identifier: Apache-2.0

// The following file is autogenerated with py2svh.py tool.

`ifndef I3C_CONFIG
`define I3C_CONFIG

  `define CMD_FIFO_DEPTH   64
  `define RX_FIFO_DEPTH    64
  `define TX_FIFO_DEPTH    64
  `define RESP_FIFO_DEPTH  64
  `define IBI_FIFO_DEPTH   64
  `define DAT_DEPTH        128
  `define DCT_DEPTH        128
  `define I3C_USE_AXI      1
  `define AXI_ADDR_WIDTH   12
  `define AXI_DATA_WIDTH   32
  `define AXI_USER_WIDTH   32
  `define AXI_ID_WIDTH     8
  `define AXI_ID_FILTERING 1
  `define NUM_PRIV_IDS     4
  `define DISABLE_INPUT_FF 1
  `define TARGET_SUPPORT   1

`endif  // I3C_CONFIG
