# Design verification

This chapter presents the available models and tools which are used for I3C verification.
The core is verified with [the Cocotb/Verilator + unit tests](https://github.com/chipsalliance/i3c-core/tree/main/verification/cocotb/block) and [the UVM test suite](https://github.com/chipsalliance/i3c-core/tree/main/verification/uvm_i3c).

## Testplans

This section contains testplans for the verification.

Definitions:
* `testplan` - an organized collection of testpoints
* `testpoint` - an actionable item, which can be turned into a test:
    * `name` - typically related to the tested feature
    * `desc` - detailed description; should contain description of the feature, configuration mode, stimuli, expected behavior.
    * `stage` - can be used to assign testpoints to milestones.
    * `tests` - names of implemented tests, which cover the testpoint. Relation test-testpoint can be many to many.
    * `tags` - additional tags that can be used to group testpoints

### Testplans for individual blocks

```{include} ../../verification/testplan/generated/bus_monitor.md
```
```{include} ../../verification/testplan/generated/bus_rx_flow.md
```
```{include} ../../verification/testplan/generated/bus_timers.md
```
```{include} ../../verification/testplan/generated/bus_tx_flow.md
```
```{include} ../../verification/testplan/generated/bus_tx.md
```
```{include} ../../verification/testplan/generated/ccc.md
```
```{include} ../../verification/testplan/generated/csr_sw_access.md
```
```{include} ../../verification/testplan/generated/descriptor_rx.md
```
```{include} ../../verification/testplan/generated/descriptor_tx.md
```
```{include} ../../verification/testplan/generated/drivers.md
```
```{include} ../../verification/testplan/generated/edge_detector.md
```
```{include} ../../verification/testplan/generated/flow_standby_i3c.md
```
```{include} ../../verification/testplan/generated/hci_queues.md
```
```{include} ../../verification/testplan/generated/tti_queues.md
```
```{include} ../../verification/testplan/generated/i3c_bus_monitor.md
```
```{include} ../../verification/testplan/generated/pec.md
```
```{include} ../../verification/testplan/generated/width_converter_8toN.md
```
```{include} ../../verification/testplan/generated/width_converter_Nto8.md
```

### Testplans for the core

```{include} ../../verification/testplan/generated/target_ccc.md
```
```{include} ../../verification/testplan/generated/target_hdr.md
```
```{include} ../../verification/testplan/generated/target_interrupts.md
```
```{include} ../../verification/testplan/generated/target.md
```
```{include} ../../verification/testplan/generated/target_recovery.md
```
```{include} ../../verification/testplan/generated/target_reset.md
