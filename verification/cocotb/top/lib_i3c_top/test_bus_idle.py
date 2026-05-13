# SPDX-License-Identifier: Apache-2.0

import logging
import cocotb
from cocotb.triggers import ClockCycles, FallingEdge, Timer
from boot import boot_init
from i3c_controller_fixed import I3cControllerFixed as I3cController
from interface import I3CTopTestInterface
from common import log_seed
from bus2csr import int2dword


_BUS_TIMERS_PATH = (
    "xi3c_wrapper.i3c.xcontroller.xcontroller_standby"
    ".xcontroller_standby_i3c.xbus_timers"
)


async def test_setup(dut):
    cocotb.log.setLevel(logging.DEBUG)
    log_seed(dut)
    i3c_controller = I3cController(
        sda_i=dut.bus_sda, sda_o=dut.sda_sim_ctrl_i,
        scl_i=dut.bus_scl, scl_o=dut.scl_sim_ctrl_i,
        debug_state_o=None, speed=12.5e6,
    )
    i3c_controller.monitor_enable.clear()
    await i3c_controller.monitor_idle.wait()
    dut.sda_sim_target_i.value = 1
    dut.scl_sim_target_i.value = 1
    dut.peripheral_reset_done_i.value = 0
    tb = I3CTopTestInterface(dut)
    await tb.setup(fclk=333.0)
    await ClockCycles(tb.clk, 50)
    await boot_init(tb, fclk=333.0)
    return i3c_controller, tb


@cocotb.test()
async def test_bus_idle(dut):
    """
    Ensures target enters and leaves bus idle state after certain delays.
    """
    i3c_controller, tb = await test_setup(dut)
    bus_idle_sig = getattr(dut, _BUS_TIMERS_PATH + ".bus_idle_o")

    # 1. Generate a manual STOP condition to start the bus timers
    # (The timer requires a STOP detection edge to restart its internal counters)
    dut._log.info("Generating STOP condition (SDA 0->1 while SCL=1) to start bus timers")
    i3c_controller.scl = 1
    i3c_controller.sda = 0
    await Timer(2, "us")
    i3c_controller.sda = 1  # STOP edge
    await Timer(2, "us")

    # 2. Wait > 200us for T_IDLE. This forces bus_idle_o to toggle 0 -> 1.
    dut._log.info("Waiting 210us for bus_idle_o to assert")
    await Timer(210, "us")
    assert bus_idle_sig.value == 1, "Target should be in bus idle state"

    # 3. Generate a manual START condition to break the idle state (1 -> 0 toggle)
    dut._log.info("Generating START condition (SDA 1->0 while SCL=1) to deassert bus_idle_o")
    i3c_controller.sda = 0
    await Timer(2, "us")
    assert bus_idle_sig.value == 0, "Target should not be in bus idle state"

    await tb.teardown()


@cocotb.test
async def test_exotic_idle_timings(dut):
    """
    The bus conditions in the bus_timers module operate independently from each other
    with each one configured by its own CSR. This introduces the unlikely possibility
    of T_AVAL < T_FREE and T_IDLE < T_AVAL. This test exists to cover these conditions.
    """

    i3c_controller, tb = await test_setup(dut)
    T_FREE = tb.reg_map.I3C_EC.SOCMGMTIF.T_FREE_REG.base_addr
    T_AVAL = tb.reg_map.I3C_EC.SOCMGMTIF.T_AVAL_REG.base_addr
    T_IDLE = tb.reg_map.I3C_EC.SOCMGMTIF.T_IDLE_REG.base_addr
    SIG_FREE = getattr(dut, _BUS_TIMERS_PATH + ".bus_free_o")
    SIG_AVAL = getattr(dut, _BUS_TIMERS_PATH + ".bus_available_o")
    SIG_IDLE = getattr(dut, _BUS_TIMERS_PATH + ".bus_idle_o")
    SIG_BUSY = getattr(dut, _BUS_TIMERS_PATH + ".bus_busy_o")

    # COND 1: BUS AVAILABLE, BUT NOT FREE NOR IDLE
    await tb.write_csr(T_AVAL, int2dword(0x100), 4)
    await tb.write_csr(T_FREE, int2dword(0x1000), 4)
    await tb.write_csr(T_IDLE, int2dword(0x1000), 4)
    await i3c_controller.send_start()
    await i3c_controller.send_stop()
    await FallingEdge(SIG_BUSY)
    assert ~SIG_FREE.value and ~SIG_IDLE.value and SIG_AVAL.value

    # COND 2: BUS IDLE, BUT NOT AVAILABLE NOR FREE
    await tb.write_csr(T_AVAL, int2dword(0x1000), 4)
    await tb.write_csr(T_FREE, int2dword(0x1000), 4)
    await tb.write_csr(T_IDLE, int2dword(0x100), 4)
    await i3c_controller.send_start()
    await i3c_controller.send_stop()
    await FallingEdge(SIG_BUSY)
    assert ~SIG_FREE.value and SIG_IDLE.value and ~SIG_AVAL.value

    await tb.teardown()
