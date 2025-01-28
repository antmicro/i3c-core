## target_interrupts

### Testpoints

##### `rx_desc_stat`

Test: `rx_desc_stat`

Enables RX_DESC_STAT TTI interrupt, checks if the irq_o signal is
deasserted, sends a private write over I3C to the target and
waits for irq_o assertion. Once the interrupt is asserted reads
a RX descriptor from the TTI RX descriptor queue, ensures that
irq_o gets deasserted after the read.

##### `tx_desc_stat`

Test: `tx_desc_stat`

Enables TX_DESC_STAT TTI interrupt, checks if the irq_o signal is
deasserted, writes data to TTI TX data queue followed by writing
a descriptor to TTI TX descriptor queue, sends a private read
over I3C and waits for irq_o assertion. Once the interrupt is
asserted clears it by writing 1 to the TX_DESC_STAT fiels of TTI
INTERRUPT_STATUS csr and ensures that irq_o signal gets deasserted.

##### `ibi_done`

Test: `ibi_done`

Enables IBI_DONE_EN TTI interrupt, checks if the irq_o signal is
deasserted, and the status bit in TTI INTERRUPT_STATUS CSR cleared.
Issues and IBI, waits for it to be serviced by the controller.
Checks if the status bit is set in INTERRUPT_STATUS CSR and the
irq_o signal asserted. Reads LAST_IBI_STATUS field from the TTI
STATUS CSR, ensures that irq_o gets deasserted and status bit gets
cleared afterwards.

##### `interrupt_force`

Test: `interrupt_force`

The test is run for each TTI interrupt:
 - TX_DESC_STAT_EN
 - RX_DESC_STAT_EN
 - RX_DESC_THLD_STAT_EN
 - RX_DATA_THLD_STAT_EN
 - IBI_DONE_EN

Ensures that irq_o is deasserted. Disables the interrupt in TTI
INTERRUPT_ENABLE CSR, forces the interrupt by writing 1 to the
corresponding field in TTI INTERRUPT_FORCE CSR, ensures that
the irq_o does not get asserted.

Enables the interrupt in TTI INTERRUPT_ENABLE CSR, forces the
interrupt by writing 1 to the corresponding field in
TTI INTERRUPT_FORCE CSR, ensures that the irq_o does get asserted.

Clears the interrupt by writing 1 to its corresponding field in
TTI INTERRUPT_STATUS CSR, ensures that irq_o gets deasserted and
the status bit cleared.


