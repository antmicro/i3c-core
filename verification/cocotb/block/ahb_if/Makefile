# SPDX-License-Identifier: Apache-2.0

null  :=
space := $(null) #
comma := ,

TEST_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
SRC_DIR := $(abspath $(TEST_DIR)../../../../../src)

TEST_FILES   = $(sort $(wildcard test_*.py))

# Set appropriate bus interface via Cocotb's PLUSARGS:
override PLUSARGS := $(strip +FrontendBusInterface=AHB $(PLUSARGS))

MODULE      ?= $(subst $(space),$(comma),$(subst .py,,$(TEST_FILES)))
TOPLEVEL     = ahb_if_wrapper
CFG_NAME     = ahb_target_only

VERILOG_SOURCES  = \
    $(SRC_DIR)/i3c_pkg.sv \
    $(CALIPTRA_ROOT)/src/libs/rtl/ahb_defines_pkg.sv \
    $(SRC_DIR)/csr/controller_I3CCSR_pkg.sv \
    $(SRC_DIR)/csr/controller_I3CCSR.sv \
    $(SRC_DIR)/csr/target_I3CCSR_pkg.sv \
    $(SRC_DIR)/csr/target_I3CCSR.sv \
    $(SRC_DIR)/csr/controller_and_target_I3CCSR_pkg.sv \
    $(SRC_DIR)/csr/controller_and_target_I3CCSR.sv \
    $(SRC_DIR)/hci/ahb_if.sv \
    $(TEST_DIR)/ahb_if_wrapper.sv

include $(TEST_DIR)/../block_common.mk
