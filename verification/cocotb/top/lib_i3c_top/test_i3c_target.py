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


def format_ibi_data(mdb, data):
    """
    Given MDB and a list of data bytes (can be empty) prepare a sequence of
    32-bit words to be written to the TTI IBI queue.
    """
    count = (len(data) + 3) // 4
    words = [0 for i in range(count)]

    i = 0
    j = 0
    for d in data:
        words[j] |= d << (8 * i)

        i = i + 1
        if i == 4:
            i = 0
            j = j + 1

    descr = (mdb << 24) | len(data)
    return [descr] + words


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


@cocotb.test()
async def test_i3c_target_ibi(dut):
    """
    IBI test. Sends an IBI with no data and then subsequently IBIs with
    different data lengths. Expects the controller to ACK all of them and
    return correctly received data.
    """

    # Setup
    i3c_controller, i3c_target, tb = await test_setup(dut)

    target = i3c_controller.add_target(TARGET_ADDRESS)
    target.set_bcr_fields(ibi_req_capable=True, ibi_payload=True)

    result = True

    # Enable IBI ACK-ing
    i3c_controller.enable_ibi(True)

    # Write descriptor to the TTI IBI queue. No IBI data
    mdb = 0xAA
    data = []
    ibi_data = format_ibi_data(mdb, data)
    dut._log.info(" ".join([f"0x{d:08X}" for d in ibi_data]))
    for word in ibi_data:
        await tb.write_csr(tb.reg_map.I3C_EC.TTI.IBI_PORT.base_addr, int2dword(word), 4)

    # Wait for the IBI to be serviced, check data
    response = await i3c_controller.wait_for_ibi()
    expected = bytearray([TARGET_ADDRESS, mdb] + data)
    if response != expected:
        dut._log.critical(
            "IBI MDB/data mismatch! tgt: [ {}] ctl: [ {}]".format(
                "".join("".join(f"0x{d:02X}") + " " for d in expected),
                "".join("".join(f"0x{d:02X}") + " " for d in response),
            )
        )
        result = False

    # Check LAST_IBI_STATUS
    status = dword2int(await tb.read_csr(tb.reg_map.I3C_EC.TTI.STATUS.base_addr, 4))
    last_ibi_status = (status & (3 << 14)) >> 14
    expected_status = 0
    if last_ibi_status != expected_status:
        dut._log.critical(
            f"Incorrect IBI status, expected {expected_status}, got {last_ibi_status}"
        )
        result = False

    await ClockCycles(tb.clk, 50)

    # Write descriptor to the TTI IBI queue with some data. Check different
    # data lengths to exercise 32-bit to 8-bit conversion that happens inside
    # IBI module
    payload = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xCA]

    for lnt in [4, 5, 6, 7, 8]:

        mdb = 0xAA
        data = payload[: lnt + 1]
        ibi_data = format_ibi_data(mdb, data)
        dut._log.info(" ".join([f"0x{d:08X}" for d in ibi_data]))
        for word in ibi_data:
            await tb.write_csr(tb.reg_map.I3C_EC.TTI.IBI_PORT.base_addr, int2dword(word), 4)

        # Wait for the IBI to be serviced, check data
        response = await i3c_controller.wait_for_ibi()
        expected = bytearray([TARGET_ADDRESS, mdb] + data)
        if response != expected:
            dut._log.critical(
                "IBI MDB/data mismatch! tgt: [ {}] ctl: [ {}]".format(
                    "".join("".join(f"0x{d:02X}") + " " for d in expected),
                    "".join("".join(f"0x{d:02X}") + " " for d in response),
                )
            )
            result = False

        # Check LAST_IBI_STATUS
        status = dword2int(await tb.read_csr(tb.reg_map.I3C_EC.TTI.STATUS.base_addr, 4))
        last_ibi_status = (status & (3 << 14)) >> 14
        expected_status = 0
        if last_ibi_status != expected_status:
            dut._log.critical(
                f"Incorrect IBI status, expected {expected_status}, got {last_ibi_status}"
            )
            result = False

        await ClockCycles(tb.clk, 50)

    # Report the test result
    assert result


@cocotb.test()
async def test_i3c_target_ibi_retry(dut):
    """
    Disables IBI ACK-ing in controller, sends an IBI, waits some time for the
    target to retry IBI transmission, re-enables IBI-acking, waits until the
    IBI gets serviced, check if IBI data was received correctly.
    """

    # Setup
    i3c_controller, i3c_target, tb = await test_setup(dut)

    # Enable indefinite IBI retries
    #  TTI.CONTROL.IBI_EN        = 1
    #  TTI.CONTROL.IBI_RETRY_NUM = 7 (means indefinite)
    await tb.write_csr(tb.reg_map.I3C_EC.TTI.CONTROL.base_addr, int2dword(0x0000F000), 4)

    target = i3c_controller.add_target(TARGET_ADDRESS)
    target.set_bcr_fields(ibi_req_capable=True, ibi_payload=True)

    result = True

    # Disable IBI ACK-ing
    i3c_controller.enable_ibi(False)

    # Write descriptor to the TTI IBI queue
    mdb = 0xAA
    data = [0xBE, 0xEF]
    ibi_data = format_ibi_data(mdb, data)
    dut._log.info(" ".join([f"0x{d:08X}" for d in ibi_data]))
    for word in ibi_data:
        await tb.write_csr(tb.reg_map.I3C_EC.TTI.IBI_PORT.base_addr, int2dword(word), 4)

    # Wait for some time so that the target gets a change to retry IBI
    # transmission
    await Timer(5, "us")

    # Check LAST_IBI_STATUS
    status = dword2int(await tb.read_csr(tb.reg_map.I3C_EC.TTI.STATUS.base_addr, 4))
    last_ibi_status = (status & (3 << 14)) >> 14
    expected_status = 3
    if last_ibi_status != expected_status:
        dut._log.critical(
            f"Incorrect IBI status, expected {expected_status}, got {last_ibi_status}"
        )
        result = False

    # Re-enable IBI ACK-ing
    i3c_controller.enable_ibi(True)

    # Wait for the IBI to be serviced, check data
    response = await i3c_controller.wait_for_ibi()
    expected = bytearray([TARGET_ADDRESS, mdb] + data)
    if response != expected:
        dut._log.critical(
            "IBI MDB/data mismatch! tgt: [ {}] ctl: [ {}]".format(
                "".join("".join(f"0x{d:02X}") + " " for d in expected),
                "".join("".join(f"0x{d:02X}") + " " for d in response),
            )
        )
        result = False

    # Check LAST_IBI_STATUS
    status = dword2int(await tb.read_csr(tb.reg_map.I3C_EC.TTI.STATUS.base_addr, 4))
    last_ibi_status = (status & (3 << 14)) >> 14
    expected_status = 0
    if last_ibi_status != expected_status:
        dut._log.critical(
            f"Incorrect IBI status, expected {expected_status}, got {last_ibi_status}"
        )
        result = False

    # Dummy wait
    await ClockCycles(tb.clk, 10)

    # Report the test result
    assert result


@cocotb.test()
async def test_i3c_target_ibi_data(dut):
    """
    Set a limit on how many IBI data bytes the controller may accept. Issue
    an IBI with more data and check if it gets serviced correctly. Finally
    issue yet another IBI to check if target logic flushed the remaining data
    correctly.
    """

    # Setup
    i3c_controller, i3c_target, tb = await test_setup(dut)

    target = i3c_controller.add_target(TARGET_ADDRESS)
    target.set_bcr_fields(ibi_req_capable=True, ibi_payload=True)

    result = True

    # Limit IBI data count that the controller can accept
    i3c_controller.set_max_ibi_data_len(6)

    # Write descriptor to the TTI IBI queue
    mdb = 0xAA
    data = [0xCA, 0xFE, 0xBA, 0xCA, 0xAA, 0xBB, 0xCC, 0xDD]
    ibi_data = format_ibi_data(mdb, data)
    dut._log.info(" ".join([f"0x{d:08X}" for d in ibi_data]))
    for word in ibi_data:
        await tb.write_csr(tb.reg_map.I3C_EC.TTI.IBI_PORT.base_addr, int2dword(word), 4)

    # Wait for the IBI to be serviced, check data
    response = await i3c_controller.wait_for_ibi()
    expected = bytearray([TARGET_ADDRESS, mdb] + data[:6])
    if response != expected:
        dut._log.critical(
            "IBI MDB/data mismatch! tgt: [ {}] ctl: [ {}]".format(
                "".join("".join(f"0x{d:02X}") + " " for d in expected),
                "".join("".join(f"0x{d:02X}") + " " for d in response),
            )
        )
        result = False

    # Wait
    await ClockCycles(tb.clk, 50)

    # Do another IBI to check if remaining data from the TTI IBI queue got
    # flushed correctly.
    mdb = 0xAA
    data = [0x11, 0x22, 0x33]
    ibi_data = format_ibi_data(mdb, data)
    dut._log.info(" ".join([f"0x{d:08X}" for d in ibi_data]))
    for word in ibi_data:
        await tb.write_csr(tb.reg_map.I3C_EC.TTI.IBI_PORT.base_addr, int2dword(word), 4)

    # Wait for the IBI to be serviced, check data
    response = await i3c_controller.wait_for_ibi()
    expected = bytearray([TARGET_ADDRESS, mdb] + data)
    if response != expected:
        dut._log.critical(
            "IBI MDB/data mismatch! tgt: [ {}] ctl: [ {}]".format(
                "".join("".join(f"0x{d:02X}") + " " for d in expected),
                "".join("".join(f"0x{d:02X}") + " " for d in response),
            )
        )
        result = False

    # Dummy wait
    await ClockCycles(tb.clk, 10)

    # Report the test result
    assert result
