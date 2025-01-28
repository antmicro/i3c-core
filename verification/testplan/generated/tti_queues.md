## tti_queues

### Testpoints

##### `tti_tx_capacity_status_test`

Test: `test_tti_tx_capacity_status_test`

Resets the TTI TX queue and verifies that it is empty
afterwards

##### `tti_tx_desc_capacity_status_test`

Test: `test_tti_tx_desc_capacity_status_test`

Resets the TTI TX descriptor queue and verifies that it is empty
afterwards

##### `tti_rx_capacity_status_test`

Test: `test_tti_rx_capacity_status_test`

Resets the TTI RX queue and verifies that it is empty
afterwards

##### `tti_rx_desc_capacity_status_test`

Test: `test_tti_rx_desc_capacity_status_test`

Resets the TTI RX descriptor queue and verifies that it is empty
afterwards

##### `tti_tx_setup_threshold_test`

Test: `test_tti_tx_setup_threshold_test`

Writes the threshold to appropriate register for the TTI data TX
queue (QUEUE_THLD_CTRL or DATA_BUFFER_THLD_CTRL).
Verifies that an appropriate value has been written to the CSR.
Verifies the threshold signal assumes the correct value.

##### `tti_tx_desc_setup_threshold_test`

Test: `test_tti_tx_desc_setup_threshold_test`

Writes the threshold to appropriate register for the TTI descriptor TX
queue (QUEUE_THLD_CTRL or DATA_BUFFER_THLD_CTRL).
Verifies that an appropriate value has been written to the CSR.
Verifies the threshold signal assumes the correct value.

##### `tti_rx_setup_threshold_test`

Test: `test_tti_rx_setup_threshold_test`

Writes the threshold to appropriate register for the TTI data RX
queue (QUEUE_THLD_CTRL or DATA_BUFFER_THLD_CTRL).
Verifies that an appropriate value has been written to the CSR.
Verifies the threshold signal assumes the correct value.

##### `tti_rx_desc_setup_threshold_test`

Test: `test_tti_rx_desc_setup_threshold_test`

Writes the threshold to appropriate register for the TTI descriptor RX
queue (QUEUE_THLD_CTRL or DATA_BUFFER_THLD_CTRL).
Verifies that an appropriate value has been written to the CSR.
Verifies the threshold signal assumes the correct value.

##### `tti_ibi_setup_threshold_test`

Test: `test_tti_ibi_setup_threshold_test`

Writes the threshold to appropriate register for the TTI IBI
queue (QUEUE_THLD_CTRL or DATA_BUFFER_THLD_CTRL).
Verifies that an appropriate value has been written to the CSR.
Verifies the threshold signal assumes the correct value.

##### `rx_desc_should_raise_thld_trig`

Test: `test_rx_desc_should_raise_thld_trig`

Sets up a ready threshold of the read queue and checks whether the
trigger signal is properly asserted at different levels of the
queue fill.

##### `rx_should_raise_thld_trig`

Test: `test_rx_should_raise_thld_trig`

Sets up a ready and start thresholds of the read queue and checks
whether the trigger signals are properly asserted at different
levels of the queue fill.

##### `tx_desc_should_raise_thld_trig`

Test: `test_tx_desc_should_raise_thld_trig`

Sets up a ready and start threshold of the write queue and checks
whether the trigger is properly asserted at different levels of
the queue fill.

##### `tx_should_raise_thld_trig`

Test: `test_tx_should_raise_thld_trig`

Sets up a ready and start threshold of the write queue and checks
whether the trigger is properly asserted at different levels of
the queue fill.

##### `ibi_should_raise_thld_trig`

Test: `test_ibi_should_raise_thld_trig`

Sets up a ready and start threshold of the write queue and checks
whether the trigger is properly asserted at different levels of
the queue fill.


