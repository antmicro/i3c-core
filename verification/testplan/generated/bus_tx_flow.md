## bus_tx_flow

### Testpoints

##### `bit_tx_negedge`

Test: `test_bit_tx_negedge`

Requests the bus_tx_flow module to drive SDA right after SCL falling
edge. Checks if the requested bit value is driven correctly

##### `bit_tx_pre_posedge`

Test: `test_bit_tx_pre_posedge`

Requests the bus_tx_flow module to drive SDA just before SCL rising
edge. Checks if the requested bit value is driven correctly

##### `bit_tx_high_level`

Test: `test_bit_tx_high_level`

Requests the bus_tx_flow module to drive SDA just before SCL falling
edge. Checks if the requested bit value is driven correctly

##### `bit_tx_low_level`

Test: `test_bit_tx_low_level`

Requests the bus_tx_flow module to drive SDA when SCL in in stable
low state. Checks if the requested bit value is driven correctly

##### `byte_tx`

Test: `test_byte_tx`

Requests the bus_tx_flow module to transmitt a data byte along with
T-bit. While the transmission is in progress samples SDA on rising
edges of SCL. Once the transmission finishes compares sampled data
with what was requested to be sent.


