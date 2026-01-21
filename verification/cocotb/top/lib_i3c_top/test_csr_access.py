# SPDX-License-Identifier: Apache-2.0

from itertools import chain, product
import logging
from math import log2
import random

from bus2csr import bytes2int, compare_values, int2dword
from interface import I3CTopTestInterface

import cocotb
from cocotb_helpers import reset_n
from cocotb.triggers import RisingEdge, Timer
from cocotbext.axi import AxiLockType, AxiBurstType, AxiResp


async def timeout_task(timeout):
    await Timer(timeout, "us")
    raise RuntimeError("Test timeout!")


async def initialize(dut, fclk=333.0, timeout=50):
    """
    Common test initialization routine
    """

    cocotb.log.setLevel(logging.DEBUG)

    # Start the background timeout task
    await cocotb.start(timeout_task(timeout))

    tb = I3CTopTestInterface(dut)
    await tb.setup(fclk)
    return tb


async def adjust_queue_thld_to_boundary(tb, reg_if, name, wdata):
    # Name of the queue & its respective queue size field name
    fields = {
        "tx_desc": "TX_DESC_BUFFER_SIZE",
        "rx_desc": "RX_DESC_BUFFER_SIZE",
        "ibi": "IBI_QUEUE_SIZE",
    }

    reg = getattr(reg_if, f"{'IBI_' if name == 'ibi' else ''}QUEUE_SIZE")
    field = getattr(reg, fields[name])
    qsize = bytes2int(await tb.read_csr(reg.base_addr, 4))

    # All sizes are stored as n where the size in DWORDs is calculated as 2 ** (n + 1)
    qsize = 2 ** (((qsize & field.mask) >> field.low) + 1)

    # IBI size is not adjusted by design
    if name == "ibi":
        return wdata
    return min(wdata, qsize - 1)


def rand_reg_val(reg, is_unhandled=False):
    wdata = random.randint(0, 2**32 - 1)
    exp_rd = 0
    for f_name in reg:
        if f_name in ["base_addr", "offset"]:
            continue
        f = getattr(reg, f_name)
        reset = 0 if "reset" not in f else f.reset
        if f.sw == "r" or is_unhandled:
            data = (reset << f.low) & f.mask
        elif any([f.woclr, f.hwclr, f.sw == "w"]):
            data = 0
        else:
            data = wdata & f.mask
        exp_rd |= data
    return wdata, exp_rd


def csr_access_test_data(reg_if, skip_regs=[]):
    """
    reg_if: dict
        Sub-dictionary of `common.reg_map`. Contains a collection of registers.
        Each register contains a collection of register fields.
    skip_regs: list
        Names of the registers to be excluded from generated test data.

    Takes a dictionary of registers and prepares CSR read-write test data.
    Draws a random 32-bit word and deduces expected read from the register based on
    register field descriptions.

    Will expect to read `0` if register is set to clear on write or contains `hwclr` property.

    Will expect to read `reset` value (`0` if `reset` is not specified) if a register is read-only
    by software.

    Otherwise, will expect to read respective sub-word with account for field mask.

    NOTE: Limitation of this function is that it will only prepare test data for the registers
          at the depth `1` of the `reg_if`.
          Will skip registers that are contained within the additional regfiles of the `reg_if`.
    """
    skip_regs = skip_regs.copy()
    skip_regs.extend(["start_addr"])
    test_data = []
    for reg_name in reg_if:
        # Do not consider embedded register structures for now
        if reg_name in skip_regs or "base_addr" not in getattr(reg_if, reg_name):
            continue
        reg = getattr(reg_if, reg_name)
        test_data.append([reg_name, reg.base_addr, *rand_reg_val(reg)])
    return test_data


async def run_basic_csr_access(tb, reg_if, exceptions=[]):
    ITERS = 10
    for _ in range(ITERS):
        test_data = csr_access_test_data(reg_if, skip_regs=exceptions)
        for _, addr, wdata, exp_rd in test_data:
            await tb.write_csr(addr, int2dword(wdata), 4)
            rd_data = await tb.read_csr(addr)
            compare_values(int2dword(exp_rd), rd_data, addr)
            # TODO: Take into account read values from the CSRs and drop this reset
            await reset_n(tb.clk, tb.rst_n, cycles=2)


@cocotb.test()
async def test_basic_burst_read(dut):
    tb = await initialize(dut, timeout=500)

    # Dump the entire register space
    mem_dump = []
    for addr in range(0, 1024, 4):
        data = await tb.read_csr(addr, 4)
        mem_dump.extend(list(data))

    burst_lens = {
        AxiBurstType.FIXED: range(15),
        AxiBurstType.INCR: (1 << i for i in range(8)), # toggles each bit
        AxiBurstType.WRAP: (1, 3, 7, 15)
    }

    # Check if variously parametrized AXI burst reads yield the same values
    # This depends on all reads having no side-effect that would
    # result in a different value
    for arburst, arlens in burst_lens.items():
        for arlen, arsize, arlock, aruser in product(
            arlens, (0, 1, 2), AxiLockType, (0, 0xAAAAAAAA, 0x55555555)
        ):
            start = random.randrange(0, 600, 4)
            if arburst == AxiBurstType.WRAP:
                alignto = (2 ** arsize) * (arlen + 1)
                start &= 0xffffffff << (int(log2(alignto)))
            bursted = await tb.busIf.axi_m.read(
                start,
                size=arsize,
                length=arlen+1,
                lock=arlock,
                user=aruser,
                burst=arburst
            )
            bursted = list(bursted.data)
            # Fixed reads seem to act like INCR and return consecutive bytes
            # if ARSIZE = 0 and ARLEN > 0, instead of the same byte
            if arburst is not AxiBurstType.FIXED:
                assert bursted == mem_dump[start:start+len(bursted)]


@cocotb.test()
async def test_basic_burst_write(dut):
    # inner registers that are problematic to validate
    # e.g. have hwclr fields that don't yield 0
    exceptions = {
        "RECOVERY_STATUS",
        "DEVICE_STATUS_0",
        "STBY_CR_CONTROL",
        "STBY_CR_STATUS",
        "STBY_CR_INTR_STATUS",
        "STBY_CR_INTR_SIGNAL_ENABLE",
        "STBY_CR_INTR_FORCE",
        "STBY_CR_CCC_CONFIG_GETCAPS",
        "STBY_CR_CCC_CONFIG_RSTACT_PARAMS",
        "__RSVD_3",
        "QUEUE_THLD_CTRL",
        "RESET_CONTROL",
        "INTERRUPT_STATUS",
        "REC_INTF_REG_W1C_ACCESS",
        "INDIRECT_FIFO_CTRL_0"
    }

    tb = await initialize(dut, timeout=500)

    i3ec = tb.reg_map.I3C_EC
    filtered_regs = (
        r for r in i3ec if r != "start_addr" and "base_addr" not in getattr(i3ec, r)
    )

    # an iterator over test data for all registers in all I3C_EC.* groups
    test_data = chain(*(csr_access_test_data(getattr(i3ec, r)) for r in filtered_regs))

    write_map = [0] * 1024
    expect_map = [None] * 1024 # None values will not be verified

    for reg_name, addr, wdata, exp_rd in test_data:
        if reg_name in exceptions:
            continue
        for i, (wb, eb) in enumerate(zip(int2dword(wdata), int2dword(exp_rd))):
            write_map[addr+i] = wb
            expect_map[addr+i] = eb

    burst_lens = {
        AxiBurstType.FIXED: range(15),
        AxiBurstType.INCR: (1 << i for i in range(8)), # toggles each bit
        AxiBurstType.WRAP: (1, 3, 7, 15)
    }

    for awburst, awlens in burst_lens.items():
        for awlen, awsize, awlock, awuser in product(
            awlens, (0, 1, 2), AxiLockType, (0, 0xAAAAAAAA, 0x55555555)
        ):
            size = 2**awsize
            addr = random.randrange(0, 500, 4)
            if awburst == AxiBurstType.WRAP:
                alignto = (2 ** awsize) * (awlen + 1)
                addr &= 0xffffffff << (int(log2(alignto)))

            endaddr = addr + size * (awlen + 1)

            if awburst is AxiBurstType.FIXED:
                wdata = write_map[addr:addr+size] * (awlen + 1)
            else:
                wdata = write_map[addr:endaddr]

            resp = await tb.busIf.axi_m.write(
                addr,
                bytes(wdata),
                user=awuser,
                lock=awlock,
                size=awsize,
                wuser=awuser,
                burst=awburst
            )
            assert resp.resp == AxiResp.OKAY

            rd_dat = []
            if awburst is AxiBurstType.FIXED:
                if expect_map[addr] is None:
                    continue
                data = (await tb.read_csr(addr, 4))[0:size]
                rd_dat = list(data * (awlen + 1))
                assert (e := expect_map[addr:addr+size] * (awlen + 1)) == rd_dat, (
                    f"Sequence at {addr:#x} differs. Written: {wdata}. Expected: {e}. Got: {rd_dat}"
                )
            else:
                for adr in range(addr, endaddr, 4):
                    data = await tb.read_csr(adr, 4)
                    rd_dat.extend(list(data))

                for i in range(endaddr-addr):
                    if expect_map[addr+i] is None:
                        continue

                    assert (e := expect_map[addr+i]) == (a := rd_dat[i]), (
                        f"Byte at {addr+i:#x} differs. Written: {write_map[addr+i]:#x}. "
                        f"Expected: {e:#x}. Actual: {a:#x}"
                    )

            await reset_n(tb.clk, tb.rst_n, cycles=2)


@cocotb.test(skip=("ControllerSupport" not in cocotb.plusargs))
async def test_dat_csr_access(dut):
    tb = await initialize(dut)
    await run_basic_csr_access(tb, tb.reg_map.DAT)


@cocotb.test(skip=("ControllerSupport" not in cocotb.plusargs))
async def test_dct_csr_access(dut):
    exceptions = [
        "DCT_MEMORY",  # Out-of-use
    ]
    tb = await initialize(dut)
    await run_basic_csr_access(tb, tb.reg_map.DCT, exceptions)


@cocotb.test(skip=("ControllerSupport" not in cocotb.plusargs))
async def test_base_csr_access(dut):
    exceptions = [
        "RESET_CONTROL",
    ]
    tb = await initialize(dut)
    await run_basic_csr_access(tb, tb.reg_map.I3CBASE, exceptions)

    # RESET_CONTROL
    val, exp_rd = rand_reg_val(tb.reg_map.I3CBASE.RESET_CONTROL)
    # SOFT_RST field is supported (doesn't trigger clears; is not cleared itself)
    exp_rd |= val & tb.reg_map.I3CBASE.RESET_CONTROL.SOFT_RST.mask
    addr = tb.reg_map.I3CBASE.RESET_CONTROL.base_addr
    await tb.write_csr(addr, int2dword(val), 4)

    # The queues are empty and should proceed with reset immediately
    await RisingEdge(tb.clk)
    rd_data = await tb.read_csr(addr)
    compare_values(int2dword(exp_rd), rd_data, addr)


@cocotb.test(skip=("ControllerSupport" not in cocotb.plusargs))
async def test_pio_csr_access(dut):
    exceptions = [
        "RESPONSE_PORT",
        "TX_DATA_PORT",
        "RX_DATA_PORT",
        "IBI_PORT",
        "QUEUE_THLD_CTRL",
    ]
    tb = await initialize(dut)
    await run_basic_csr_access(tb, tb.reg_map.PIOCONTROL, exceptions)


@cocotb.test()
async def test_ec_sec_fw_rec_csr_access(dut):
    exceptions = [
        "INDIRECT_FIFO_CTRL_0",
        "INDIRECT_FIFO_DATA",  # Viable only in recovery mode
    ]
    tb = await initialize(dut)
    await run_basic_csr_access(tb, tb.reg_map.I3C_EC.SECFWRECOVERYIF, exceptions)


@cocotb.test()
async def test_ec_stdby_ctrl_mode_csr_access(dut):
    exceptions = [
        "STBY_CR_CONTROL",
        "STBY_CR_STATUS",
        "STBY_CR_INTR_STATUS",
        "STBY_CR_INTR_SIGNAL_ENABLE",
        "STBY_CR_INTR_FORCE",
        "STBY_CR_CCC_CONFIG_GETCAPS",
        "STBY_CR_CCC_CONFIG_RSTACT_PARAMS",
        "__RSVD_3",
    ]

    tb = await initialize(dut)
    await run_basic_csr_access(tb, tb.reg_map.I3C_EC.STDBYCTRLMODE, exceptions)

    # Standby Controller Mode CSRs that are not supported or are reserved
    for reg_name in exceptions:
        reg = getattr(tb.reg_map.I3C_EC.STDBYCTRLMODE, reg_name)
        addr = reg.base_addr
        exp_rd, val = 0, 0

        reg_mask = 2**32 - 1
        # Fields without 'we' are overwritten by hw
        if reg_name == "STBY_CR_CONTROL":
            val, exp_rd = rand_reg_val(reg)
            hw_w_fields = [
                "PENDING_RX_NACK",
                "HANDOFF_DELAY_NACK",
                "ACR_FSM_OP_SELECT",
                "PRIME_ACCEPT_GETACCCR",
            ]
            sticky_wr = ["HANDOFF_DEEP_SLEEP"]
            for f_name in hw_w_fields:
                field = getattr(tb.reg_map.I3C_EC.STDBYCTRLMODE.STBY_CR_CONTROL, f_name)
                exp_rd &= reg_mask - field.mask
            for f_name in sticky_wr:
                field = getattr(tb.reg_map.I3C_EC.STDBYCTRLMODE.STBY_CR_CONTROL, f_name)
                exp_rd |= 1 << field.low
        elif reg_name == "STBY_CR_CCC_CONFIG_RSTACT_PARAMS":
            val, exp_rd = rand_reg_val(reg)
            hw_w_fields = ["RESET_TIME_PERIPHERAL", "RESET_TIME_TARGET"]
            for f_name in hw_w_fields:
                field = getattr(tb.reg_map.I3C_EC.STDBYCTRLMODE.STBY_CR_CCC_CONFIG_RSTACT_PARAMS, f_name)
                exp_rd &= reg_mask - field.mask
        elif reg_name == "STBY_CR_INTR_STATUS":
            val, exp_rd = rand_reg_val(reg)
            hw_w_fields = [
                "ACR_HANDOFF_OK_REMAIN_STAT",
                "ACR_HANDOFF_OK_PRIMED_STAT",
                "ACR_HANDOFF_ERR_FAIL_STAT",
                "ACR_HANDOFF_ERR_M3_STAT",
                "CRR_RESPONSE_STAT",
                "STBY_CR_DYN_ADDR_STAT",
                "STBY_CR_ACCEPT_NACKED_STAT",
                "STBY_CR_ACCEPT_OK_STAT",
                "STBY_CR_ACCEPT_ERR_STAT",
                "CCC_PARAM_MODIFIED_STAT",
                "CCC_UNHANDLED_NACK_STAT",
                "CCC_FATAL_RSTDAA_ERR_STAT",
            ]
            for f_name in hw_w_fields:
                field = getattr(tb.reg_map.I3C_EC.STDBYCTRLMODE.STBY_CR_INTR_STATUS, f_name)
                exp_rd &= reg_mask - field.mask
        elif reg_name == "STBY_CR_INTR_SIGNAL_ENABLE":
            val, exp_rd = rand_reg_val(reg)
            hw_w_fields = [
                "ACR_HANDOFF_OK_REMAIN_SIGNAL_EN",
                "ACR_HANDOFF_OK_PRIMED_SIGNAL_EN",
                "ACR_HANDOFF_ERR_FAIL_SIGNAL_EN",
                "ACR_HANDOFF_ERR_M3_SIGNAL_EN",
                "CRR_RESPONSE_SIGNAL_EN",
                "STBY_CR_DYN_ADDR_SIGNAL_EN",
                "STBY_CR_ACCEPT_NACKED_SIGNAL_EN",
                "STBY_CR_ACCEPT_OK_SIGNAL_EN",
                "STBY_CR_ACCEPT_ERR_SIGNAL_EN",
                "CCC_PARAM_MODIFIED_SIGNAL_EN",
                "CCC_UNHANDLED_NACK_SIGNAL_EN",
                "CCC_FATAL_RSTDAA_ERR_SIGNAL_EN",
            ]
            for f_name in hw_w_fields:
                field = getattr(tb.reg_map.I3C_EC.STDBYCTRLMODE.STBY_CR_INTR_SIGNAL_ENABLE, f_name)
                exp_rd &= reg_mask - field.mask
        else:
            val, _ = rand_reg_val(reg, is_unhandled=True)

        await tb.write_csr(addr, int2dword(val), 4)
        rd_data = await tb.read_csr(addr)
        compare_values(int2dword(exp_rd), rd_data, addr)


@cocotb.test()
async def test_ec_tti_csr_access(dut):
    exceptions = [
        "RESET_CONTROL",
        "RX_DESC_QUEUE_PORT",
        "RX_DATA_PORT",
        "QUEUE_THLD_CTRL",
    ]
    tb = await initialize(dut)
    await run_basic_csr_access(tb, tb.reg_map.I3C_EC.TTI, exceptions)

    # Target Transaction Interface CSRs that implement custom behavior
    # or are not supported / reserved
    for reg_name in exceptions:
        wait = None
        reg_mask = 2**32 - 1
        reg = getattr(tb.reg_map.I3C_EC.TTI, reg_name)
        addr = reg.base_addr
        exp_rd, val = 0, 0

        if reg_name == "RESET_CONTROL":
            val, exp_rd = rand_reg_val(reg)
            # SOFT_RST field is supported (doesn't trigger clears; is not cleared itself)
            exp_rd |= val & tb.reg_map.I3C_EC.TTI.RESET_CONTROL.SOFT_RST.mask
            # The queues are empty and should proceed with reset immediately
            wait = RisingEdge(tb.clk)
        elif reg_name == "QUEUE_THLD_CTRL":
            val, exp_rd = rand_reg_val(reg)
            fields = {
                "TX_DESC_THLD": "tx_desc",
                "RX_DESC_THLD": "rx_desc",
                "IBI_THLD": "ibi",
            }
            for f_name, qname in fields.items():
                field = getattr(tb.reg_map.I3C_EC.TTI.QUEUE_THLD_CTRL, f_name)
                new_thld = (val & field.mask) >> field.low

                q_rd = await adjust_queue_thld_to_boundary(tb, tb.reg_map.I3C_EC.TTI, qname, new_thld)
                exp_rd &= reg_mask - field.mask  # clear
                exp_rd |= q_rd << field.low
        else:
            val, _ = rand_reg_val(reg, is_unhandled=True)

        await tb.write_csr(addr, int2dword(val), 4)
        if wait:
            await wait
        rd_data = await tb.read_csr(addr)
        compare_values(int2dword(exp_rd), rd_data, addr)


@cocotb.test()
async def test_ec_soc_mgmt_csr_access(dut):
    exceptions = ["REC_INTF_REG_W1C_ACCESS"]
    tb = await initialize(dut)
    await run_basic_csr_access(tb, tb.reg_map.I3C_EC.SOCMGMTIF, exceptions)


@cocotb.test()
async def test_ec_contrl_config_csr_access(dut):
    tb = await initialize(dut)
    await run_basic_csr_access(tb, tb.reg_map.I3C_EC.CTRLCFG)


@cocotb.test()
async def test_ec_csr_access(dut):
    tb = await initialize(dut)
    await run_basic_csr_access(tb, tb.reg_map.I3C_EC)
