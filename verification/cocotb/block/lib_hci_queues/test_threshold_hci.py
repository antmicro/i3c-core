# SPDX-License-Identifier: Apache-2.0

from common_methods import (
    setup_sim,
    should_setup_ready_threshold,
    CmdQueueThldHandler,
    should_setup_start_threshold,
    RxQueueThldHandler,
    TxQueueThldHandler,
    RespQueueThldHandler,
    IbiQueueThldHandler,
    should_raise_start_thld_trig_receiver,
    should_raise_ready_thld_trig_receiver,
    should_raise_ready_thld_trig_transmitter,
    should_raise_start_thld_trig_transmitter,
)

from cocotb.handle import SimHandleBase
from utils import controller_test


@controller_test()
async def test_cmd_setup_threshold(dut: SimHandleBase):
    interface = await setup_sim(dut, "hci")
    await should_setup_ready_threshold(interface, CmdQueueThldHandler())


@controller_test()
async def test_rx_setup_threshold(dut: SimHandleBase):
    interface = await setup_sim(dut, "hci")
    await should_setup_start_threshold(interface, RxQueueThldHandler())
    await should_setup_ready_threshold(interface, RxQueueThldHandler())


@controller_test()
async def test_tx_setup_threshold(dut: SimHandleBase):
    interface = await setup_sim(dut, "hci")
    await should_setup_start_threshold(interface, TxQueueThldHandler())
    await should_setup_ready_threshold(interface, TxQueueThldHandler())


@controller_test()
async def test_resp_setup_threshold(dut: SimHandleBase):
    interface = await setup_sim(dut, "hci")
    await should_setup_ready_threshold(interface, RespQueueThldHandler())


@controller_test()
async def test_ibi_setup_threshold(dut: SimHandleBase):
    interface = await setup_sim(dut, "hci")
    await should_setup_ready_threshold(interface, IbiQueueThldHandler())


@controller_test()
async def test_resp_should_raise_thld_trig(dut: SimHandleBase):
    interface = await setup_sim(dut, "hci")
    await should_raise_ready_thld_trig_receiver(interface, RespQueueThldHandler())


@controller_test()
async def test_rx_should_raise_thld_trig(dut: SimHandleBase):
    interface = await setup_sim(dut, "hci")
    await should_raise_start_thld_trig_receiver(interface, RxQueueThldHandler())
    await should_raise_ready_thld_trig_receiver(interface, RxQueueThldHandler())


@controller_test()
async def test_ibi_should_raise_thld_trig(dut: SimHandleBase):
    interface = await setup_sim(dut, "hci")
    await should_raise_ready_thld_trig_receiver(interface, IbiQueueThldHandler())


@controller_test()
async def test_cmd_should_raise_thld_trig(dut: SimHandleBase):
    interface = await setup_sim(dut, "hci")
    await should_raise_ready_thld_trig_transmitter(interface, CmdQueueThldHandler())


@controller_test()
async def test_tx_should_raise_thld_trig(dut: SimHandleBase):
    interface = await setup_sim(dut, "hci")
    await should_raise_start_thld_trig_transmitter(interface, TxQueueThldHandler())
    await should_raise_ready_thld_trig_transmitter(interface, TxQueueThldHandler())
