# SPDX-License-Identifier: Apache-2.0

TOPLEVEL_LANG    = verilog
SIM             ?= verilator
WAVES           ?= 1

# Paths
CURDIR = $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
I3C_ROOT := $(abspath $(CURDIR)/../..)
CFGDIR :=
CONFIG :=
$(info From common.mk, CURDIR is $(CURDIR))

# Set pythonpath so that tests can access common modules
export PYTHONPATH := $(PYTHONPATH):$(CURDIR)/common

VERILOG_INCLUDE_DIRS= \
    $(CALIPTRA_ROOT)/src/libs/rtl \
    $(CALIPTRA_ROOT)/src/caliptra_prim/rtl \
    $(I3C_ROOT)/src \
    $(I3C_ROOT)/src/libs/axi

$(info VERILOG_SOURCES = $(VERILOG_SOURCES))
VERILOG_SOURCES := $(COMMON_SOURCES) $(VERILOG_SOURCES)
$(info VERILOG_SOURCES = $(VERILOG_SOURCES))

# Coverage reporting
COVERAGE_TYPE ?=
ifeq ("$(COVERAGE_TYPE)", "all")
    VERILATOR_COVERAGE = --coverage
else ifeq ("$(COVERAGE_TYPE)", "branch")
    VERILATOR_COVERAGE = --coverage-line
else ifeq ("$(COVERAGE_TYPE)", "toggle")
    VERILATOR_COVERAGE = --coverage-toggle
else ifeq ("$(COVERAGE_TYPE)", "functional")
    VERILATOR_COVERAGE = --coverage-user
else
    VERILATOR_COVERAGE = ""
endif

# Enable processing of #delay statements
ifeq ($(SIM), verilator)
    COMPILE_ARGS += --timing
    COMPILE_ARGS += +define+DIGITAL_IO_I3C
    COMPILE_ARGS += -Wall -Wno-fatal
    COMPILE_ARGS += --x-assign unique --x-initial unique

    EXTRA_ARGS += --trace --trace-structs
    EXTRA_ARGS += $(VERILATOR_COVERAGE)
    EXTRA_ARGS += -Wno-DECLFILENAME -Wno-TIMESCALEMOD
endif

ifeq ($(SIM), vcs)
    COMPILE_ARGS += +libext+.sv +libext+.v
    COMPILE_ARGS += +define+DIGITAL_IO_I3C
    COMPILE_ARGS += $(foreach dir,$(VERILOG_INCLUDE_DIRS),-y $(dir))
    COMPILE_ARGS += -debug_access+all +memcbk
    SIM_ARGS += +dumpon
    EXTRA_ARGS += +vcs+vcdpluson +vpdfile+dump.vpd

    ifneq ($(COVERAGE_TYPE),)
        EXTRA_ARGS += -cm line+cond+fsm+tgl+branch
    endif
endif

COCOTB_HDL_TIMEUNIT         = 1ns
COCOTB_HDL_TIMEPRECISION    = 10ps

# Build directory
comma := ,
ifneq ($(COVERAGE_TYPE),)
    # Check if more than one test is provided
    ifeq ($(findstring $(comma),$(MODULE)),$(comma))
        # To collect accurate coverage results each tests needs to have a unique SIM_BUILD directory to store
        # the results. If multiple tests were to use the same directory they would override each others coverage reports
        # causing the reported values to be incorrect.
        $(error Collecting coverage for multiple tests is not supported. Either unset 'COVERAGE_TYPE' to run tests without coverage reporting or use nox.)
    else
        # Construct a unique directory for each test and coverage type
        SIM_BUILD := sim_build-$(MODULE)-$(COVERAGE_TYPE)
    endif
endif

include $(shell cocotb-config --makefiles)/Makefile.sim

ifeq ($(SIM), vcs)

.PHONY: convert-vpd2vcd
convert-vpd2vcd: $(COCOTB_RESULTS_FILE)
	vpd2vcd -full64 dump.vpd dump.vcd +splitpacked

all: sim convert-vpd2vcd

endif
