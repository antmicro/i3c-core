## ccc

### Testpoints

##### `get_status`

Test: `test_get_status`

Instucts the ccc module to begin servicing GETSTATUS CCC. Feeds
data bytes and bits to the module via its bus_tx/bus_rx interfaces
to mimick actual I3C transaction. Checks if data bytes received
correspond to correct GETSTATUS CCC response.


