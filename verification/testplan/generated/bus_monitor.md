## bus_monitor

### Testpoints

##### `get_status`

Test: `test_get_status`

Tests operation of the bus_monitor module along with its sub-modules.
Performs a number of I3C transactions between a simulated controller
and a simulated target. Counts start, repeated start and stop events
reported by bus_monitor. Verifies that the counts match what's expected.


