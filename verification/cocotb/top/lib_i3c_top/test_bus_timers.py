# SPDX-License-Identifier: Apache-2.0

"""
Bus timers top-level tests.

Verifies that the bus_timers module correctly:
  - Starts the counter on STOP conditions
  - Resets the counter on START/RSTART conditions
  - Resets the counter when entering HDR mode (in_hdr_mode_i held high)
"""

import logging
import random

from boot import boot_init
from bus2csr import int2dword
from i3c_controller_fixed import I3cControllerFixed as I3cController
from interface import I3CTopTestInterface

import cocotb
from cocotb.triggers import ClockCycles

from common import VALID_I3C_ADDRESSES, log_seed

# =============================================================================
# Constants
# =============================================================================

ENTHDR0 = 0x20

# Small timer thresholds so tests run quickly (unit: DUT clock cycles)
T_FREE = 5
T_AVAL = 15
T_IDLE = 30

# Hierarchy path to the xbus_timers instance
BUS_TIMERS_PATH = (
    "xi3c_wrapper.i3c.xcontroller.xcontroller_standby"
    ".xcontroller_standby_i3c.xbus_timers"
)

# Hierarchy path to the target FSM state register
FSM_STATE_PATH = (
    "xi3c_wrapper.i3c.xcontroller.xcontroller_standby"
    ".xcontroller_standby_i3c.xi3c_target_fsm.state_q"
)

FSM_STATE_IDLE = 0
FSM_STATE_IN_HDR_MODE = 19


# =============================================================================
# Helpers
# =============================================================================

async def test_setup(dut, dynamic_addr=None):
    """Set up controller and top-level core interface.

    No external target BFM is needed for bus timer tests; the DUT itself
    processes all bus conditions.
    """
    cocotb.log.setLevel(logging.DEBUG)
    log_seed(dut)

    i3c_controller = I3cController(
        sda_i=dut.bus_sda,
        sda_o=dut.sda_sim_ctrl_i,
        scl_i=dut.bus_scl,
        scl_o=dut.scl_sim_ctrl_i,
        debug_state_o=None,
        speed=12.5e6,
    )

    dut.sda_sim_target_i.setimmediatevalue(1)
    dut.scl_sim_target_i.setimmediatevalue(1)

    tb = I3CTopTestInterface(dut)
    await tb.setup()
    await ClockCycles(tb.clk, 50)
    await boot_init(tb, static_addr=0x5A, virtual_static_addr=0x5B,
                    dynamic_addr=dynamic_addr)
    return i3c_controller, tb


async def set_timer_thresholds(tb, t_free, t_aval, t_idle):
    """Override bus timer CSR thresholds with test-specific values."""
    await tb.write_csr(
        tb.reg_map.I3C_EC.SOCMGMTIF.T_FREE_REG.base_addr, int2dword(t_free), 4)
    await tb.write_csr(
        tb.reg_map.I3C_EC.SOCMGMTIF.T_AVAL_REG.base_addr, int2dword(t_aval), 4)
    await tb.write_csr(
        tb.reg_map.I3C_EC.SOCMGMTIF.T_IDLE_REG.base_addr, int2dword(t_idle), 4)


def _bt_sig(dut, name):
    """Return a bus_timers signal handle by name, using a full cocotb path."""
    return getattr(dut, BUS_TIMERS_PATH + "." + name)


def read_bus_state(dut):
    """Return current bus timer outputs and counter value as a dict."""
    return {
        "counter":   int(_bt_sig(dut, "bus_state_counter").value),
        "busy":      int(_bt_sig(dut, "bus_busy_o").value),
        "free":      int(_bt_sig(dut, "bus_free_o").value),
        "available": int(_bt_sig(dut, "bus_available_o").value),
        "idle":      int(_bt_sig(dut, "bus_idle_o").value),
    }


def get_fsm_state(dut):
    return int(getattr(dut, FSM_STATE_PATH).value)


# =============================================================================
# Tests
# =============================================================================

@cocotb.test()
async def test_bus_timers_stop_starts_counter(dut):
    """STOP condition starts the bus timer counter; states advance correctly.

    Verifies the counter transitions through busy → free → available → idle
    after a STOP condition is detected.  Before any STOP the counter must be
    at 0 with bus_busy asserted.
    """
    i3c_controller, tb = await test_setup(dut)
    await set_timer_thresholds(tb, T_FREE, T_AVAL, T_IDLE)

    # Before STOP: counter=0, bus_busy=1 (count_enable has not been set yet)
    state = read_bus_state(dut)
    assert state["counter"] == 0, \
        f"Expected counter=0 before STOP, got {state['counter']}"
    assert state["busy"] == 1, \
        f"Expected bus_busy=1 before STOP, got {state['busy']}"
    assert state["free"] == 0, \
        f"Expected bus_free=0 before STOP, got {state['free']}"

    # Drive START then STOP on the bus to start the timer
    await i3c_controller.send_start()
    await i3c_controller.send_stop()

    # Wait long enough for all thresholds to be passed
    await ClockCycles(tb.clk, T_IDLE + 20)

    state = read_bus_state(dut)
    assert state["free"] == 1, \
        f"Expected bus_free=1 after {T_FREE} cycles, got {state['free']}"
    assert state["available"] == 1, \
        f"Expected bus_available=1 after {T_AVAL} cycles, got {state['available']}"
    assert state["idle"] == 1, \
        f"Expected bus_idle=1 after {T_IDLE} cycles, got {state['idle']}"
    assert state["busy"] == 0, \
        f"Expected bus_busy=0 when idle, got {state['busy']}"

    await tb.teardown()


@cocotb.test()
async def test_bus_timers_reset_on_start(dut):
    """START condition resets the bus timer counter to 0.

    After a STOP has started the counter and bus_free has been observed,
    a subsequent START must immediately reset the counter back to 0 and
    deassert all timer-state outputs.
    """
    i3c_controller, tb = await test_setup(dut)
    await set_timer_thresholds(tb, T_FREE, T_AVAL, T_IDLE)

    # Start the counter with a STOP condition
    await i3c_controller.send_start()
    await i3c_controller.send_stop()

    # Wait until bus_free fires
    await ClockCycles(tb.clk, T_FREE + 10)
    state = read_bus_state(dut)
    assert state["free"] == 1, \
        f"Expected bus_free=1 before START reset, " \
        f"got {state['free']} (counter={state['counter']})"

    # START condition resets the counter and clears count_enable
    await i3c_controller.send_start()
    await ClockCycles(tb.clk, 3)

    state = read_bus_state(dut)
    assert state["counter"] == 0, \
        f"Expected counter=0 after START, got {state['counter']}"
    assert state["free"] == 0, \
        f"Expected bus_free=0 after START reset, got {state['free']}"
    assert state["available"] == 0, \
        f"Expected bus_available=0 after START reset, got {state['available']}"
    assert state["busy"] == 1, \
        f"Expected bus_busy=1 after START reset, got {state['busy']}"

    # Close the open frame
    await i3c_controller.send_stop()
    await tb.teardown()


@cocotb.test()
async def test_bus_timers_reset_on_hdr_entry(dut):
    """Entering HDR mode holds the bus timer counter at 0.

    The bus_timers RTL drives reset_counter = in_hdr_mode_i, so while
    the target FSM is in HDR mode the counter is continuously cleared.
    After exiting HDR mode the counter stays at 0 until the next STOP.
    """
    DYNAMIC_ADDR = random.choice(VALID_I3C_ADDRESSES)
    i3c_controller, tb = await test_setup(dut, dynamic_addr=DYNAMIC_ADDR)
    await set_timer_thresholds(tb, T_FREE, T_AVAL, T_IDLE)

    # Step 1: start the counter with a STOP condition
    await i3c_controller.send_start()
    await i3c_controller.send_stop()
    await ClockCycles(tb.clk, T_FREE + 10)

    state = read_bus_state(dut)
    assert state["free"] == 1, \
        f"Expected bus_free=1 before HDR entry, " \
        f"got {state['free']} (counter={state['counter']})"

    # Step 2: enter HDR mode via ENTHDR0 broadcast CCC
    # This also drives a START condition (resetting the counter), then
    # asserts in_hdr_mode_i which keeps reset_counter=1 throughout HDR mode.
    await i3c_controller.i3c_ccc_write(
        ENTHDR0, broadcast_data=[], stop=False, pull_scl_low=True)
    await ClockCycles(tb.clk, 10)
    assert get_fsm_state(dut) == FSM_STATE_IN_HDR_MODE, \
        f"Expected FSM in HDR mode ({FSM_STATE_IN_HDR_MODE}), " \
        f"got {get_fsm_state(dut)}"

    # Step 3: counter must stay at 0 while in HDR mode
    await ClockCycles(tb.clk, T_IDLE + 20)
    state = read_bus_state(dut)
    assert state["counter"] == 0, \
        f"Expected counter=0 in HDR mode, got {state['counter']}"
    assert state["busy"] == 1, \
        f"Expected bus_busy=1 in HDR mode, got {state['busy']}"
    assert state["free"] == 0, \
        f"Expected bus_free=0 in HDR mode, got {state['free']}"
    assert get_fsm_state(dut) == FSM_STATE_IN_HDR_MODE, \
        f"Expected FSM still in HDR mode, got {get_fsm_state(dut)}"

    # Step 4: exit HDR mode; counter starts counting (STOP condition is send as a part of HDR exit)
    await i3c_controller.send_hdr_exit()
    await ClockCycles(tb.clk, 5)
    assert get_fsm_state(dut) == FSM_STATE_IDLE, \
        f"Expected FSM idle after HDR exit, got {get_fsm_state(dut)}"

    state = read_bus_state(dut)
    assert state["counter"] != 0, \
        f"Expected counter!=0 after HDR exit (no STOP yet), got {state['counter']}"

    assert state["free"] == 1, \
        f"Expected bus_free=1 after STOP following HDR exit, got {state['free']}"

    await tb.teardown()
