## hci_queues

### Testpoints

##### `clear_on_nonempty_resp_queue`

Test: `test_clear_on_nonempty_resp_queue`

Writes to the HCI queue RESET_CONTROL CSR bit which causes HCI
command response queue to be cleared. Then, polls the CSR until the
bit gets cleared by the hardware. To check if the queue has been
cleared puts a descriptor to the queue and reads it back. It
should be the same descriptor.

##### `clear_on_nonempty_cmd_queue`

Test: `test_clear_on_nonempty_cmd_queue`

Puts a command descriptor to the HCI command queue. Writes to the
RESET_CONTROL CSR to the bit responsible for clearing the queue,
polls the CSR until the bit gets cleared by hardware. Verifies that
the queue got cleared by pushing and retrieving another descriptor
from the queue.

##### `clear_on_nonempty_rx_queue`

Test: `test_clear_on_nonempty_rx_queue`

Puts 10 data words to the HCI RX data queue. Writes to the
RESET_CONTROL CSR to the bit responsible for clearing the queue,
polls the CSR until the bit gets cleared by hardware. Puts and
gets another data word from the queue to check if it was cleared

##### `clear_on_nonempty_tx_queue`

Test: `test_clear_on_nonempty_tx_queue`

Puts 10 data words to the HCI TX data queue. Writes to the
RESET_CONTROL CSR to the bit responsible for clearing the queue,
polls the CSR until the bit gets cleared by hardware. Puts and
gets another data word from the queue to check if it was cleared

##### `clear_on_nonempty_ibi_queue`

Test: `test_clear_on_nonempty_ibi_queue`

Puts 10 data words to the HCI IBI queue. Writes to the
RESET_CONTROL CSR to the bit responsible for clearing the queue,
polls the CSR until the bit gets cleared by hardware. Puts and
gets another data word from the queue to check if it was cleared

##### `cmd_capacity_status_test`

Test: `test_cmd_capacity_status_test`

Resets the HCI command queue and verifies that it is empty
afterwards

##### `resp_capacity_status_test`

Test: `test_resp_capacity_status_test`

Resets the HCI response queue and verifies that it is empty
afterwards

##### `rx_capacity_status_test`

Test: `test_rx_capacity_status_test`

Resets the HCI RX queue and verifies that it is empty
afterwards

##### `tx_capacity_status_test`

Test: `test_tx_capacity_status_test`

Resets the HCI TX queue and verifies that it is empty
afterwards

##### `ibi_capacity_status_test`

Test: `test_ibi_capacity_status_test`

Resets the HCI IBI queue and verifies that it is empty
afterwards

##### `cmd_setup_threshold_test`

Test: `test_cmd_setup_threshold_test`

Writes the threshold to appropriate register for the HCI command
queue (QUEUE_THLD_CTRL or DATA_BUFFER_THLD_CTRL).
Verifies that an appropriate value has been written to the CSR.
Verifies the threshold signal assumes the correct value.

##### `resp_setup_threshold_test`

Test: `test_resp_setup_threshold_test`

Writes the threshold to appropriate register for the HCI response
queue (QUEUE_THLD_CTRL or DATA_BUFFER_THLD_CTRL).
Verifies that an appropriate value has been written to the CSR.
Verifies the threshold signal assumes the correct value.

##### `rx_setup_threshold_test`

Test: `test_rx_setup_threshold_test`

Writes the threshold to appropriate register for the HCI data RX
queue (QUEUE_THLD_CTRL or DATA_BUFFER_THLD_CTRL).
Verifies that an appropriate value has been written to the CSR.
Verifies the threshold signal assumes the correct value.

##### `tx_setup_threshold_test`

Test: `test_tx_setup_threshold_test`

Writes the threshold to appropriate register for the HCI data TX
queue (QUEUE_THLD_CTRL or DATA_BUFFER_THLD_CTRL).
Verifies that an appropriate value has been written to the CSR.
Verifies the threshold signal assumes the correct value.

##### `ibi_setup_threshold_test`

Test: `test_ibi_setup_threshold_test`

Writes the threshold to appropriate register for the HCI IBI
queue (QUEUE_THLD_CTRL or DATA_BUFFER_THLD_CTRL).
Verifies that an appropriate value has been written to the CSR.
Verifies the threshold signal assumes the correct value.

##### `resp_should_raise_thld_trig`

Test: `test_resp_should_raise_thld_trig`

Sets up a ready threshold of the read queue and checks whether the
trigger signal is properly asserted at different levels of the
queue fill.

##### `rx_should_raise_thld_trig`

Test: `test_rx_should_raise_thld_trig`

Sets up a ready and start thresholds of the read queue and checks
whether the trigger signals are properly asserted at different
levels of the queue fill.

##### `ibi_should_raise_thld_trig`

Test: `test_ibi_should_raise_thld_trig`

Sets up a ready threshold of the read queue and checks whether the
trigger signal is properly asserted at different levels of the
queue fill.

##### `cmd_should_raise_thld_trig`

Test: `test_cmd_should_raise_thld_trig`

Sets up a ready threshold of the write queue and checks whether
the trigger is properly asserted at different levels of the queue
fill.

##### `tx_should_raise_thld_trig`

Test: `test_tx_should_raise_thld_trig`

Sets up a ready and start threshold of the write queue and checks
whether the trigger is properly asserted at different levels of
the queue fill.


