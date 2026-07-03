# tags
# - "tests"
# - "ahb"
# - "axi"
# - "axi_fast"
# - "axi_block"
#
# sessions
#
# > All sessions are named after items in the testplan, with "_verify" suffixed to their names.

import functools
import os
import random
import time
import shutil

from dataclasses import dataclass, field
from typing import List

import nox
from nox_utils import VerificationTest, isCocotbSimFailure, nox_config, sim_repeater_path

# Common nox configuration
nox = nox_config(nox)

# Test configuration
pip_requirements_path = "../../requirements.txt"

simulators = [os.getenv("SIMULATOR", "verilator")]

# Coverage types to collect
if os.getenv("TEST_COVERAGE_ENABLE", "0") == "1":
    coverage_types = ["vcs"] if "vcs" in simulators else ["all"]
else:
    coverage_types = None

i3c_root = os.getenv("I3C_ROOT_DIR")

dut_config = os.getenv("DUT_CONFIG", "target_only")

@dataclass
class TestParams:
    tags: List[str]
    test_group: List[str]
    test_name: List[str]
    coverage: None | List[str] = field(
        default_factory=lambda: coverage_types.copy() if coverage_types else None
    )
    simulator: List[str] = field(default_factory=lambda: simulators.copy())


def test(params: TestParams):
    def wrapper(func):
        # Skip tests that require a feature not present in the current DUT_CONFIG
        if "target" in params.tags and dut_config == "controller_only":
            return
        if "controller" in params.tags and dut_config == "target_only":
            return

        # Apply parametrize decorators
        for k, v in reversed(params.__dict__.items()):
            if k != "tags":
                func = nox.parametrize(k, v)(func)

        session_decorator = nox.session(tags=params.tags) if params.tags else nox.session()

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)

        return session_decorator(wrapped)

    return wrapper

def _verify(session, test_group, test_type, test_name, coverage=None, simulator=None):
    test_iterations = int(os.getenv("TEST_ITERATIONS", 1))

    for i in range(test_iterations):
        pfx = "" if test_iterations == 1 else f"_{i}"
        test = VerificationTest(test_group, test_type, test_name, coverage, pfx)
        
        # Translate session options to plusargs
        plusargs = list(session.posargs)

        # Randomize seed for initialization of undefined signals in the simulation
        random.seed(time.time_ns())
        seed = random.randint(1, 10000)

        with open(test.paths["log_default"], "w") as test_log:
            # Remove simulation build artifacts
            if simulator == "vcs" and i > 0:
                shutil.rmtree(os.path.join(test.testPath, test.sim_build))

            args = [
                sim_repeater_path(),
                "make",
                "-C",
                test.testPath,
                "all",
                "MODULE=" + test_name,
                "COCOTB_RESULTS_FILE=" + test.filenames["xml"],
            ]
            if test_type == "top":
                args.append("DUT_CONFIG=" + dut_config)

            if simulator == "verilator":
                plusargs.extend([f"+verilator+seed+{seed}"])
                if os.getenv("WAVES", "0") == "1":
                    plusargs.append("--trace")
                    
            if coverage:
                args.append("COVERAGE_TYPE=" + coverage)

            if simulator:
                args.append("SIM=" + simulator)

            args.append("PLUSARGS=" + " ".join(plusargs))

            session.run(
                *args,
                external=True,
                stdout=test_log,
                stderr=test_log,
            )
            
        # Prevent coverage.dat and test log from being overwritten
        test.rename_defaults(coverage, simulator)

        # Add check from results.xml to notify nox that test failed
        if isCocotbSimFailure(resultsFile=test.paths["xml"]):
            raise Exception("SimFailure: cocotb failed. See test logs for more information.")


def verify_block(session, test_group, test_name, coverage=None, simulator=None):
    _verify(session, test_group, "block", test_name, coverage, simulator)


def verify_top(session, test_group, test_name, coverage=None, simulator=None):
    _verify(session, test_group, "top", test_name, coverage, simulator)


@nox.session(tags=["tests", "ahb"])
@nox.parametrize("test_group", ["ahb_if"])
@nox.parametrize("test_name", ["test_csr_sw_access"])
@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def ahb_if_verify(session, test_group, test_name, coverage, simulator):
    verify_block(session, test_group, test_name, coverage, simulator)


@nox.session(tags=["tests", "axi", "axi_block"])
@nox.parametrize("test_group", ["axi_adapter"])
@nox.parametrize("test_name", ["test_csr_sw_access", "test_bus_stress"])
@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def axi_adapter_verify(session, test_group, test_name, coverage, simulator):
    verify_block(session, test_group, test_name, coverage, simulator)


@nox.session(tags=["tests", "axi", "axi_block"])
@nox.parametrize("test_group", ["axi_adapter_id_filter"])
@nox.parametrize("test_name", ["test_seq_csr_access", "test_bus_stress", "test_priv_id_variation"])
@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def axi_adapter_id_filter_verify(session, test_group, test_name, coverage, simulator):
    verify_block(session, test_group, test_name, coverage, simulator)


@nox.session(tags=["tests", "ahb", "axi", "axi_block"])
@nox.parametrize("test_group", ["bus_rx_flow"])
@nox.parametrize("test_name", ["test_bus_rx_flow"])
@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def bus_rx_flow_verify(session, test_group, test_name, coverage, simulator):
    verify_block(session, test_group, test_name, coverage, simulator)


@nox.session(tags=["tests", "ahb", "axi", "axi_block"])
@nox.parametrize("test_group", ["bus_tx_flow"])
@nox.parametrize("test_name", ["test_bus_tx_flow"])
@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def bus_tx_flow_verify(session, test_group, test_name, coverage, simulator):
    verify_block(session, test_group, test_name, coverage, simulator)


@nox.session(tags=["tests", "ahb"])
@nox.parametrize("test_group", ["hci_queues_ahb"])
@nox.parametrize("test_name", ["test_clear", "test_empty", "test_read_write_ports", "test_threshold"])
@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def hci_queues_ahb_verify(session, test_group, test_name, coverage, simulator):
    verify_block(session, test_group, test_name, coverage, simulator)


@test(
    TestParams(
        ["tests", "axi", "target", "controller"],
        ["flow_active"],
        ["test_flow_active_immediate_write"],
    )
)
def flow_active_immediate_write_verify(session, test_group, test_name, coverage, simulator):
    verify_block(session, test_group, test_name, coverage, simulator)


@test(
    TestParams(
        ["tests", "axi", "target", "controller"],
        ["hci_queues_axi"],
        ["test_clear", "test_empty", "test_read_write_ports", "test_threshold"],
    )
)
@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def hci_queues_axi_verify(session, test_group, test_name, coverage, simulator):
    verify_block(session, test_group, test_name, coverage, simulator)


@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def i2c_verify(session, test_group, test_name, coverage, simulator):
    verify_block(session, test_group, test_name, coverage, simulator)


@nox.session(tags=["tests", "i2c"])
@nox.parametrize("test_group", ["i2c_controller_fsm"])
@nox.parametrize("test_name", ["test_mem_rw"])
@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def i2c_controller_fsm_verify(session, test_group, test_name, coverage, simulator):
    verify_block(session, test_group, test_name, coverage, simulator)


@nox.session(tags=["tests", "i2c"])
@nox.parametrize("test_group", ["i2c_standby_controller"])
@nox.parametrize("test_name", ["test_read", "test_wr_restart_rd"])
@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def i2c_standby_controller_verify(session, test_group, test_name, coverage, simulator):
    verify_block(session, test_group, test_name, coverage, simulator)


@nox.session(tags=["tests", "i2c"])
@nox.parametrize("test_group", ["flow_standby_i2c"])
@nox.parametrize("test_name", ["test_flow_standby_i2c"])
@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def flow_standby_i2c_verify(session, test_group, test_name, coverage, simulator):
    verify_block(session, test_group, test_name, coverage, simulator)


@nox.session(tags=["tests", "i2c"])
@nox.parametrize("test_group", ["i2c_target_fsm"])
@nox.parametrize("test_name", ["test_mem_w", "test_mem_r"])
@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def i2c_target_fsm_verify(session, test_group, test_name, coverage, simulator):
    verify_block(session, test_group, test_name, coverage, simulator)


@nox.session(tags=["tests", "ahb"])
@nox.parametrize("test_group", ["i3c_ahb"])
@nox.parametrize(
    "test_name",
    [
        "test_i3c_target", "test_recovery", "test_interrupts", "test_enter_exit_hdr_mode",
        "test_bus_stall", "test_bus_timers", "test_target_reset", "test_ccc", "test_csr_access",
        "test_bypass", "test_empty_queue_read", "test_ibi", "test_ibi_multi_queue",
        "test_te_errors", "test_tsco_violation", "test_interrupt_toggles",
    ],
)
@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def i3c_ahb_verify(session, test_group, test_name, coverage, simulator):
    verify_top(session, test_group, test_name, coverage, simulator)


@nox.session(tags=["tests", "axi"])
@nox.parametrize("test_group", ["i3c_axi"])
@nox.parametrize(
    "test_name",
    [
        "test_i3c_target", "test_recovery", "test_interrupts", "test_enter_exit_hdr_mode",
        "test_bus_stall", "test_bus_timers", "test_target_reset", "test_ccc", "test_csr_access",
        "test_bypass", "test_empty_queue_read", "test_ibi", "test_ibi_multi_queue",
        "test_te_errors", "test_tsco_violation", "test_interrupt_toggles",
    ],
)
@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def i3c_axi_verify(session, test_group, test_name, coverage, simulator):
    verify_top(session, test_group, test_name, coverage, simulator)


@test(
    TestParams(
        ["tests", "axi", "target"],
        ["i3c_axi"],
        ["test_i3c_target"],
    )
)
def i3c_axi_target_verify(session, test_group, test_name, coverage, simulator):
    verify_top(session, test_group, test_name, coverage, simulator)


@test(
    TestParams(
        ["tests", "axi", "target"],
        ["i3c_axi"],
        ["test_recovery"],
    )
)
def i3c_axi_recovery_verify(session, test_group, test_name, coverage, simulator):
    verify_top(session, test_group, test_name, coverage, simulator)


@test(
    TestParams(
        ["tests", "axi", "controller"],
        ["i3c_axi_controller"],
        ["test_configure_i3c_cores"],
    )
)
def configure_i3c_cores_verify(session, test_group, test_name, coverage, simulator):
    verify_top(session, test_group, test_name, coverage, simulator)


@test(
    TestParams(
        ["tests", "axi", "controller"],
        ["i3c_axi_controller"],
        ["test_i3c_controller"],
    )
)
def i3c_controller_verify(session, test_group, test_name, coverage, simulator):
    verify_top(session, test_group, test_name, coverage, simulator)


@test(
    TestParams(
        ["tests", "axi", "controller"],
        ["i3c_axi_controller"],
        ["test_i3c_controller_write_target_read"],
    )
)
def i3c_controller_write_target_read_verify(session, test_group, test_name, coverage, simulator):
    verify_top(session, test_group, test_name, coverage, simulator)


@test(
    TestParams(
        ["tests", "axi", "controller"],
        ["i3c_axi_controller"],
        ["test_i3c_controller_repeated_start"],
    )
)
def i3c_controller_repeated_start_verify(session, test_group, test_name, coverage, simulator):
    verify_top(session, test_group, test_name, coverage, simulator)


@test(
    TestParams(
        ["tests", "axi", "controller"],
        ["i3c_axi_controller"],
        ["test_i3c_controller_read_target_write"],
    )
)
def i3c_controller_read_target_write_verify(session, test_group, test_name, coverage, simulator):
    verify_top(session, test_group, test_name, coverage, simulator)


@test(
    TestParams(
        ["tests", "axi", "controller"],
        ["i3c_axi_controller"],
        ["test_controller_ccc"],
    )
)
def controller_ccc_verify(session, test_group, test_name, coverage, simulator):
    verify_top(session, test_group, test_name, coverage, simulator)


@test(
    TestParams(
        ["tests", "axi", "controller"],
        ["i3c_axi_controller"],
        ["test_controller_hdr_exit"],
    )
)
def controller_hdr_exit_verify(session, test_group, test_name, coverage, simulator):
    verify_top(session, test_group, test_name, coverage, simulator)
    

@test(
    TestParams(
        ["tests", "axi", "controller"],
        ["i3c_axi_controller"],
        ["test_controller_ibi"],
    )
)
def controller_ibi_verify(session, test_group, test_name, coverage, simulator):
    verify_top(session, test_group, test_name, coverage, simulator)


@test(
    TestParams(
        ["tests", "axi", "controller"],
        ["i2c_axi_controller"],
        ["test_i2c_controller"],
    )
)
def i2c_controller_verify(session, test_group, test_name, coverage, simulator):
    verify_top(session, test_group, test_name, coverage, simulator)


@test(
    TestParams(
        ["tests", "axi", "controller"],
        ["i3c_axi_controller_err"],
        ["test_controller_error"],
    )
)
def controller_error_verify(session, test_group, test_name, coverage, simulator):
    verify_top(session, test_group, test_name, coverage, simulator)


@test(
    TestParams(
        ["tests", "ahb", "axi", "target"],
        ["ccc"],
        ["test_ccc"],
    )
)
@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def ccc_verify(session, test_group, test_name, coverage, simulator):
    verify_block(session, test_group, test_name, coverage, simulator)


@nox.session(tags=["tests", "ahb", "axi", "axi_block"])
@nox.parametrize("test_group", ["ctrl_bus_timers"])
@nox.parametrize("test_name", ["test_bus_timers"])
@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def ctrl_bus_timers_verify(session, test_group, test_name, coverage, simulator):
    verify_block(session, test_group, test_name, coverage, simulator)


@nox.session(tags=["tests", "ahb", "axi", "axi_block"])
@nox.parametrize("test_group", ["ctrl_bus_monitor"])
@nox.parametrize("test_name", ["test_bus_monitor"])
@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def ctrl_bus_monitor_verify(session, test_group, test_name, coverage, simulator):
    verify_block(session, test_group, test_name, coverage, simulator)
    

@nox.session(tags=["tests", "ahb", "axi", "axi_block"])
@nox.parametrize("test_group", ["ctrl_i3c_bus_monitor"])
@nox.parametrize("test_name", ["test_i3c_bus_monitor"])
@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def ctrl_i3c_bus_monitor_verify(session, test_group, test_name, coverage, simulator):
    verify_block(session, test_group, test_name, coverage, simulator)


@nox.session(tags=["tests", "ahb", "axi", "axi_block"])
@nox.parametrize("test_group", ["ctrl_edge_detector"])
@nox.parametrize("test_name", ["test_edge_detector"])
@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def ctrl_edge_detector_verify(session, test_group, test_name, coverage, simulator):
    verify_block(session, test_group, test_name, coverage, simulator)


@nox.session(tags=["tests", "ahb", "axi", "axi_block"])
@nox.parametrize("test_group", ["ctrl_descriptor_tx"])
@nox.parametrize("test_name", ["test_descriptor_tx"])
@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def ctrl_descriptor_tx_verify(session, test_group, test_name, coverage, simulator):
    verify_block(session, test_group, test_name, coverage, simulator)


@nox.session(tags=["tests", "ahb", "axi", "axi_block"])
@nox.parametrize("test_group", ["ctrl_descriptor_rx"])
@nox.parametrize("test_name", ["test_descriptor_rx"])
@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def ctrl_descriptor_rx_verify(session, test_group, test_name, coverage, simulator):
    verify_block(session, test_group, test_name, coverage, simulator)


@nox.session(reuse_venv=True)
def lint(session: nox.Session) -> None:
    session.run("isort", ".", "../../tools")
    session.run("black", "--config=pyproject.toml", ".", "../../tools")
    session.run("flake8", ".", "../../tools")


@nox.session()
def test_lint(session: nox.Session) -> None:
    session.run("isort", "--check", ".", "../../tools")
    session.run("black", "--config=pyproject.toml", "--check", ".", "../../tools")
    session.run("flake8", ".", "../../tools")


@nox.session(tags=["tests", "ahb", "axi", "axi_block"])
@nox.parametrize("test_group", ["width_converter_Nto8"])
@nox.parametrize("test_name", ["test_converter", "test_flush"])
@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def width_converter_Nto8_verify(session, test_group, test_name, coverage, simulator):
    verify_block(session, test_group, test_name, coverage, simulator)


@nox.session(tags=["tests", "ahb", "axi", "axi_block"])
@nox.parametrize("test_group", ["width_converter_8toN"])
@nox.parametrize("test_name", ["test_converter", "test_flush"])
@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def width_converter_8toN_verify(session, test_group, test_name, coverage, simulator):
    verify_block(session, test_group, test_name, coverage, simulator)


@nox.session(tags=["tests", "ahb", "axi", "axi_block"])
@nox.parametrize("test_group", ["recovery_pec"])
@nox.parametrize("test_name", ["test_pec"])
@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def recovery_pec_verify(session, test_group, test_name, coverage, simulator):
    verify_block(session, test_group, test_name, coverage, simulator)


@nox.session(tags=["tests", "axi_fast"])
@nox.parametrize("test_group", ["i3c_axi"])
@nox.parametrize(
    "test_name",
    [
        "test_interrupts", "test_enter_exit_hdr_mode", "test_bus_stall", "test_target_reset",
        "test_ccc", "test_csr_access", "test_bypass", "test_ibi", "test_ibi_multi_queue",
        "test_te_errors", "test_tsco_violation", "test_bus_timers",
    ],
)
@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def i3c_axi_fast_verify(session, test_group, test_name, coverage, simulator):
    verify_top(session, test_group, test_name, coverage, simulator)


@nox.session(tags=["tests", "axi_fast"])
@nox.parametrize("test_group", ["i3c_axi"])
@nox.parametrize("test_name", ["test_i3c_target", "test_empty_queue_read"])
@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def i3c_axi_fast_target_verify(session, test_group, test_name, coverage, simulator):
    verify_top(session, test_group, test_name, coverage, simulator)


@nox.session(tags=["tests", "axi_fast"])
@nox.parametrize("test_group", ["i3c_axi"])
@nox.parametrize("test_name", ["test_recovery"])
@nox.parametrize("coverage", coverage_types)
@nox.parametrize("simulator", simulators)
def i3c_axi_fast_recovery_verify(session, test_group, test_name, coverage, simulator):
    verify_top(session, test_group, test_name, coverage, simulator)
