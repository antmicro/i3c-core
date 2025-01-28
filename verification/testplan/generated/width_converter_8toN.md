## width_converter_8toN

### Testpoints

##### `converter`

Test: `test_converter`

Pushes random byte stream to the converter module. After each
byte waits at random. Simultaneously receives N-bit data words
and generates pushback (deasserts ready) at random. Verifies if
the output data matches the input.

##### `flush`

Test: `test_flush`

Feeds M bytes to the module where M is in [1, 2, 3]. Asserts the
sink_flush_i signal, receives the output word and checks if it
matches the input data.


