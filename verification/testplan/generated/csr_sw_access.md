## csr_sw_access

### Testpoints

##### `read_hci_version_csr`

Test: `test_read_hci_version_csr`

Reads the HCI version CSR and verifies its content

##### `read_pio_section_offset`

Test: `test_read_pio_section_offset`

Reads the PIO_SECTION_OFFSET CSR and verifies its content

##### `write_to_controller_device_addr`

Test: `test_write_to_controller_device_addr`

Writes to the CONTROLLER_DEVICE_ADDR CSR and verifies if the write was successful

##### `write_should_not_affect_ro_csr`

Test: `test_write_should_not_affect_ro_csr`

Writes to the HC_CAPABILITIES CSR which is read-only for software
Verifies that the write did not succeed.

##### `sequence_csr_read`

Test: `test_sequence_csr_read`

Performs a sequence of CSR reads. Verifies that each one succeeds

##### `sequence_csr_write`

Test: `test_sequence_csr_write`

Performs a sequence of CSR writes. Verifies that each one succeeds


