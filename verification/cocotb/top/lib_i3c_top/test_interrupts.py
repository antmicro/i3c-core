# SPDX-License-Identifier: Apache-2.0

import logging
import random

from boot import boot_init
from bus2csr import dword2int, int2dword
from cocotbext_i3c.i3c_controller import I3cController
from cocotbext_i3c.i3c_target import I3CTarget
from interface import I3CTopTestInterface

import cocotb
from cocotb.regression import TestFactory
from cocotb.triggers import ClockCycles, RisingEdge, Timer

# =============================================================================

TARGET_ADDRESS = 0x5A

async def timeout_task(timeout_us):
    """
    A generic task for handling test timeout. Waits a fixed amount of
    simulation time and then throws an exception.
    """
    await Timer(timeout_us, "us")
    raise TimeoutError("Timeout!")


async def test_setup(dut, timeout_us=50):
    """
    Sets up controller, target models and top-level core interface
    """

    cocotb.log.setLevel(logging.INFO)
    cocotb.start_soon(timeout_task(timeout_us))

    i3c_controller = I3cController(
        sda_i=dut.bus_sda,
        sda_o=dut.sda_sim_ctrl_i,
        scl_i=dut.bus_scl,
        scl_o=dut.scl_sim_ctrl_i,
        debug_state_o=None,
        speed=12.5e6,
    )

    i3c_target = I3CTarget(  # noqa
        sda_i=dut.bus_sda,
        sda_o=dut.sda_sim_target_i,
        scl_i=dut.bus_scl,
        scl_o=dut.scl_sim_target_i,
        debug_state_o=None,
        speed=12.5e6,
    )

    tb = I3CTopTestInterface(dut)
    await tb.setup()

    # Configure the top level
    await boot_init(tb)

    return i3c_controller, i3c_target, tb

# =============================================================================

@cocotb.test()
async def test_rx_desc_stat(dut):

    # Setup
    i3c_controller, _, tb = await test_setup(dut)
    irq = dut.xi3c_wrapper.i3c.irq_o

    # Enable the interrupt
    csr = tb.reg_map.I3C_EC.TTI.INTERRUPT_ENABLE
    await tb.write_csr_field(csr.base_addr, csr.RX_DESC_STAT_EN, 1)

    # Ensure that irq is low
    await ClockCycles(tb.clk, 10)
    assert irq.value == 0

    # Send a private write to the target
    async def i3c_task():
        data = [random.randint(0, 255) for i in range(4)]
        await i3c_controller.i3c_write(TARGET_ADDRESS, data)

    cocotb.start_soon(i3c_task())

    # Wait for the interrupt
    while irq.value == 0:
        await RisingEdge(tb.clk)

    # Read RX descriptor, the interrupt should go low
    await tb.read_csr(tb.reg_map.I3C_EC.TTI.RX_DESC_QUEUE_PORT.base_addr, 4)

    # Ensure that irq is low
    await ClockCycles(tb.clk, 10)
    assert irq.value == 0

    # Dummy wait
    await ClockCycles(tb.clk, 10)


async def test_interrupt_force(dut, fields):
    """
    Tests interrupt force and clear capability
    """

    # Setup
    i3c_controller, _, tb = await test_setup(dut, timeout_us=0.5)
    irq = dut.xi3c_wrapper.i3c.irq_o

    f_ena, f_force, f_sts = fields

    # Ensure that irq is low
    await ClockCycles(tb.clk, 10)
    assert irq.value == 0

    # Disable the interrupt
    csr = tb.reg_map.I3C_EC.TTI.INTERRUPT_ENABLE
    await tb.write_csr_field(csr.base_addr, getattr(csr, f_ena), 0)

    # Force the interrupt
    csr = tb.reg_map.I3C_EC.TTI.INTERRUPT_FORCE
    await tb.write_csr_field(csr.base_addr, getattr(csr, f_force), 1)
    await tb.write_csr_field(csr.base_addr, getattr(csr, f_force), 0)

    # Ensure that interrupt does not get asserted
    await ClockCycles(tb.clk, 20)
    assert irq.value == 0

    # Ensure that the status is 0
    csr = tb.reg_map.I3C_EC.TTI.INTERRUPT_STATUS
    sts = await tb.read_csr_field(csr.base_addr, getattr(csr, f_sts))
    assert sts == 0

    # Enable the interrupt
    csr = tb.reg_map.I3C_EC.TTI.INTERRUPT_ENABLE
    await tb.write_csr_field(csr.base_addr, getattr(csr, f_ena), 1)

    # Force the interrupt
    csr = tb.reg_map.I3C_EC.TTI.INTERRUPT_FORCE
    await tb.write_csr_field(csr.base_addr, getattr(csr, f_force), 1)
    await tb.write_csr_field(csr.base_addr, getattr(csr, f_force), 0)

    # Wait for the interrupt
    while irq.value == 0:
        await RisingEdge(tb.clk)

    # Ensure that the status is 1
    csr = tb.reg_map.I3C_EC.TTI.INTERRUPT_STATUS
    sts = await tb.read_csr_field(csr.base_addr, getattr(csr, f_sts))
    assert sts == 1

    # Clear the interrupt
    csr = tb.reg_map.I3C_EC.TTI.INTERRUPT_STATUS
    await tb.write_csr_field(csr.base_addr, getattr(csr, f_sts), 1)

    # Wait for the interrupt to go low
    while irq.value == 1:
        await RisingEdge(tb.clk)

    # Ensure that the status is 0
    csr = tb.reg_map.I3C_EC.TTI.INTERRUPT_STATUS
    sts = await tb.read_csr_field(csr.base_addr, getattr(csr, f_sts))
    assert sts == 0

    # Dummy wait
    await ClockCycles(tb.clk, 10)


tf = TestFactory(test_function=test_interrupt_force)
tf.add_option("fields", [
    ("RX_DESC_STAT_EN",         "RX_DESC_STAT_FORCE",  "RX_DESC_STAT"),
    ("RX_DESC_THLD_STAT_EN",    "RX_DESC_THLD_FORCE",  "RX_DESC_THLD_STAT"),
    ("RX_DATA_THLD_STAT_EN",    "RX_DATA_THLD_FORCE",  "RX_DATA_THLD_STAT"),
])
tf.generate_tests()

