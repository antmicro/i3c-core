## width_converter_Nto8

### Testpoints

##### `converter`

Test: `test_converter`

Pushes random N-bit word stream to the converter module. After each
word waits at random. Simultaneously receives bytes and generates
pushback (deasserts ready) at random. Verifies if the output data
matches the input.

##### `flush`

Test: `test_flush`

Feeds an N-bit word to the module. Receives M bytes where M is in
[1, 2, 3] and asserts source_flush_i. Verifies that the module
ceases to output data as expected.


