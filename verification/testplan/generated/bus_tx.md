## bus_tx

### Testpoints

##### `bit_tx_negedge`

Test: `test_bit_tx_negedge`

Requests the bus_tx module to drive SDA right after SCL falling
edge. Checks if the requested bit value is driven correctly

##### `bit_tx_pre_posedge`

Test: `test_bit_tx_pre_posedge`

Requests the bus_tx module to drive SDA just before SCL rising
edge. Checks if the requested bit value is driven correctly

##### `bit_tx_high_level`

Test: `test_bit_tx_high_level`

Requests the bus_tx module to drive SDA just before SCL falling
edge. Checks if the requested bit value is driven correctly

##### `bit_tx_low_level`

Test: `test_bit_tx_low_level`

Requests the bus_tx module to drive SDA when SCL in in stable
low state. Checks if the requested bit value is driven correctly

##### `byte_tx`

Test: `test_byte_tx`

Drives controls of the bus_tx module in a sequence which sends
a data byte plus T bit to the I3C bus. For each bit sent checks
if SDA is driven correctly and bus timings are met.


