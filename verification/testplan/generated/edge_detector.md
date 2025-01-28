## edge_detector

### Testpoints

##### `pretrigger_with_delay`

Test: `test_pretrigger_with_delay`

Triggers the edge_detector module before an edge on a bus line,
emits the edge and counts clock cycles it takes the detector
to report the presence of the edge. Verifies that the count is
equal to the programmed delay.

##### `posttrigger_with_delay`

Test: `test_posttrigger_with_delay`

Emits an edge on the bus, triggers the edge_detector module after
the edge when the bus line is high. Counts clock cycles it takes
the detector to report the edge event. The output detect signal
is asserted only if the bus line signal is stable for the
programmed delay time since the assertion of the trigger signal.
Verifies that the number of counted cycles is equal the programmed
delay.

##### `trigger_with_delay`

Test: `test_trigger_with_delay`

Triggers the edge detector and emits a rising edge on a bus line
simultaneously. Counts clock cycles it takes the detector
to report the presence of the edge. Verifies that the count is
equal to the programmed delay.

##### `pretrigger_with_no_delay`

Test: `test_pretrigger_with_no_delay`

Triggers the edge_detector module before an edge on a bus line,
emits the edge and counts clock cycles it takes the detector
to report the presence of the edge. Verifies that the count is
zero as the configured delay is also set to 0.

##### `posttrigger_with_no_delay`

Test: `test_posttrigger_with_no_delay`

Triggers the edge_detector module when a bus line is high which
is after an edge. Counts clock cycles it takes the detector
to report the presence of the edge. Verifies that the count is
zero as the configured delay is also set to 0.

##### `trigger_with_no_delay`

Test: `test_trigger_with_no_delay`

Triggers the edge detector and emits a rising edge on a bus line
simultaneously. Counts clock cycles it takes the detector
to report the presence of the edge. Verifies that the count is
zero as the configured delay is also set to 0.


