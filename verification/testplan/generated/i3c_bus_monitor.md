## i3c_bus_monitor

### Testpoints

##### `bus_monitor_hdr_exit`

Test: `test_bus_monitor_hdr_exit`

Verifies that the i3c_bus_monitor module correctly detects HDR
exit pattern. Sends the HDR exit pattern and verifies that the
module does not react - initially the bus is in SDR mode. Instructs
the module that the bus has entered HDR mode, issues the HDR exit
pattern and counts the number of times the module reported HDR
exit. Checks if it reported exactly one HDR exit event.

##### `target_reset_detection`

Test: `test_target_reset_detection`

Issues a target reset patterin to the I3C bus, verifies that the
i3c_bus_monitor correctly report it detected.


