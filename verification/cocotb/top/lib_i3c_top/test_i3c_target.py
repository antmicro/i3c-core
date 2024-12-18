# SPDX-License-Identifier: Apache-2.0

import logging
from math import ceil
from random import randint

from boot import boot_init
from bus2csr import dword2int, int2dword
from cocotbext_i3c.i3c_controller import I3cController
from cocotbext_i3c.i3c_target import I3CTarget
from interface import I3CTopTestInterface

import cocotb
from cocotb.triggers import ClockCycles, Timer

TARGET_ADDRESS = 0x5A


async def timeout_task(timeout_us=5):
    """
    A generic task for handling test timeout. Waits a fixed amount of
    simulation time and then throws an exception.
    """
    await Timer(timeout_us, "us")
    raise TimeoutError("Timeout!")


async def test_setup(dut):
    """
    Sets up controller, target models and top-level core interface
    """

    cocotb.log.setLevel(logging.INFO)
    cocotb.start_soon(timeout_task(100))

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


@cocotb.test()
async def test_i3c_target_write(dut):

    # Setup
    i3c_controller, i3c_target, tb = await test_setup(dut)

    # Send Private Write on I3C
    test_data = [[0xAA, 0x00, 0xBB, 0xCC, 0xDD], [0xDE, 0xAD, 0xBA, 0xBE]]
    for test_vec in test_data:
        await i3c_controller.i3c_write(TARGET_ADDRESS, test_vec)
        await ClockCycles(tb.clk, 10)

    # Wait for an interrupt
    wait_irq = True
    timeout = 0
    # Number of clock cycles after which we should observe an interrupt
    TIMEOUT_THRESHOLD = 50
    while wait_irq:
        timeout += 1
        await ClockCycles(tb.clk, 10)
        irq = dword2int(await tb.read_csr(tb.reg_map.I3C_EC.TTI.INTERRUPT_STATUS.base_addr, 4))
        if irq:
            wait_irq = False
            dut._log.debug(":::Interrupt was raised:::")
        if timeout > TIMEOUT_THRESHOLD:
            wait_irq = False
            dut._log.debug(":::Timeout cancelled polling:::")

    # Read data
    recv_data = []
    for test_vec in test_data:
        recv_xfer = []
        # Read RX descriptor
        r_data = dword2int(await tb.read_csr(tb.reg_map.I3C_EC.TTI.RX_DESC_QUEUE_PORT.base_addr, 4))
        desc_len = r_data & 0xFFFF
        assert len(test_vec) == desc_len, "Incorrect number of bytes in RX descriptor"
        remainder = desc_len % 4
        err_stat = r_data >> 28
        assert err_stat == 0, "Unexpected error detected"

        # Read RX data
        data_len = ceil(desc_len / 4)
        for _ in range(data_len):
            r_data = dword2int(await tb.read_csr(tb.reg_map.I3C_EC.TTI.RX_DATA_PORT.base_addr, 4))
            for k in range(4):
                recv_xfer.append((r_data >> (k * 8)) & 0xFF)

        # Remove entries that are outside of the data length
        if remainder:
            for k in range(4 - remainder):
                recv_xfer.pop()
        recv_data.append(recv_xfer)

    # Compare
    dut._log.info(
        "Comparing input [{}] and RX data [{}]".format(
            " ".join(["[ " + " ".join([f"0x{d:02X}" for d in s]) + " ]" for s in test_data]),
            " ".join(["[ " + " ".join([f"0x{d:02X}" for d in s]) + " ]" for s in recv_data]),
        )
    )
    assert test_data == recv_data

    # Dummy wait
    await ClockCycles(tb.clk, 10)


@cocotb.test()
async def test_i3c_target_read(dut):
    TEST_LENGTH = 3
    MAX_DATA_LEN = 10

    # Setup
    i3c_controller, i3c_target, tb = await test_setup(dut)

    for _ in range(TEST_LENGTH):
        data_len = randint(4, MAX_DATA_LEN)
        test_data = [randint(0, 255) for _ in range(data_len)]
        dut._log.info(
            "Generated data: [{}]".format(
                " ".join("".join(f"0x{d:02X}") + " " for d in test_data),
            )
        )

        # Write data to TTI TX FIFO
        for i in range(0, len(test_data), 4):
            await tb.write_csr(
                tb.reg_map.I3C_EC.TTI.TX_DATA_PORT.base_addr, test_data[i : i + 4], 4
            )

        # Write the TX descriptor
        await tb.write_csr(
            tb.reg_map.I3C_EC.TTI.TX_DESC_QUEUE_PORT.base_addr, int2dword(data_len), 4
        )

        # Issue a private read
        recv_data = await i3c_controller.i3c_read(TARGET_ADDRESS, len(test_data))
        recv_data = list(recv_data)

        # Compare
        dut._log.info(
            "Comparing input [ {}] and CSR data [ {}]".format(
                "".join("".join(f"0x{d:02X}") + " " for d in test_data),
                "".join("".join(f"0x{d:02X}") + " " for d in recv_data),
            )
        )
        assert test_data == recv_data

        # Dummy wait
        await ClockCycles(tb.clk, 100)
    await ClockCycles(tb.clk, 100)


# FIXME: Reenable after implementation
@cocotb.test(skip=True)
async def test_i3c_target_ibi(dut):
    # Setup
    i3c_controller, i3c_target, tb = await test_setup(dut)

    target = i3c_controller.add_target(TARGET_ADDRESS)
    target.set_bcr_fields(ibi_req_capable=True, ibi_payload=True)

    # Write MDB to Target's IBI queue
    mdb = 0xAA
    await tb.write_csr(tb.reg_map.I3C_EC.TTI.IBI_PORT.base_addr, int2dword(mdb), 4)

    # Wait for the IBI to be serviced, check data
    data = await i3c_controller.wait_for_ibi()
    expected = bytearray([TARGET_ADDRESS, mdb])
    assert data == expected

    # Dummy wait
    await ClockCycles(tb.clk, 10)
