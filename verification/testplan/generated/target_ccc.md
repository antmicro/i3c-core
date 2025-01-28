## CCC handling

### Testpoints

##### `ccc_getstatus`

Test: `ccc_getstatus`

The test reads PENDING_INTERRUPT field from the TTI INTERRUPT
status CSR. Next, it issues the GETSTATUS directed CCC to the
target. Finally it compares the interrupt status returned by the
CCC with the one read from the register.

##### `ccc_setdasa`

Test: `ccc_setdasa`

The test sets dynamic address and virtual dynamic address by
sending SETDASA CCC. Then it verifies that correct addresses have
been set by reading STBY_CR_DEVICE_ADDR CSR.

##### `ccc_rstdaa`

Test: `ccc_rstdaa`

Sets dynamic address via STBY_CR_DEVICE_ADDR CSR, then sends
RSTDAA CCC and verifies that the address got cleared.

##### `ccc_getbcr`

Test: `ccc_getbcr`

Reads BCR register content by sending GETBCR CCC and examining
returned data.

##### `ccc_getdcr`

Test: `ccc_getdcr`

Reads DCR register content by sending GETDCR CCC and examining
returned data.

##### `ccc_getmwl`

Test: `ccc_getmwl`

Reads MWL register content by sending GETMWL CCC and examining
returned data.

##### `ccc_getmrl`

Test: `ccc_getmrl`

Reads MRL register content by sending GETMWL CCC and examining
returned data.

##### `ccc_setaasa`

Test: `ccc_setaasa`

Issues the broadcast SETAASA CCC and checks if the target uses
its static address as dynamic by examining STBY_CR_DEVICE_ADDR
CSR.

##### `ccc_getpid`

Test: `ccc_getpid`

Sends the CCC to the target and examines if the returned PID
matches the expected.

##### `ccc_enec_disec_direct`

Test: `ccc_enec_disec_direct`

Sends DISEC CCC to the target and verifies that events are disabled.
Then, sends ENEC CCC to the target and checks that events are enabled.

##### `ccc_enec_disec_bcast`

Test: `ccc_enec_disec_bcast`

Sends boradcast DISEC CCC and verifies that events are disabled.
Then, sends broadcast ENEC CCC and checks that events are enabled.

##### `ccc_setmwl_direct`

Test: `ccc_setmwl_direct`

Sends directed SETMWL CCC to the target and verifies that the
register got correctly set. The check is performed by examining
relevant wires in the target DUT

##### `ccc_setmrl_direct`

Test: `ccc_setmrl_direct`

Sends directed SETMRL CCC to the target and verifies that the
register got correctly set. The check is performed by examining
relevant wires in the target DUT

##### `ccc_setmwl_bcast`

Test: `ccc_setmwl_bcast`

Sends broadcast SETMWL CCC and verifies that the
register got correctly set. The check is performed by examining
relevant wires in the target DUT

##### `ccc_setmrl_bcast`

Test: `ccc_setmrl_bcast`

Sends SETMRL CCC and verifies that the
register got correctly set. The check is performed by examining
relevant wires in the target DUT

##### `ccc_rstact_direct`

Test: `ccc_rstact_direct`

Sends directed RSTACT CCC to the target followed by reset pattern
and checks if reset action was stored correctly. The check is
done by examining DUT wires. Then, triggers target reset and
verifies that the peripheral_reset_o signal gets asserted.

##### `ccc_rstact_bcast`

Test: `ccc_rstact_bcast`

Sends directed RSTACT CCC to the target followed by reset pattern
and checks if reset action was stored correctly. The check is
done by examining DUT wires. Then, triggers target reset and
verifies that the escalated_reset_o signal gets asserted.

##### `ccc_direct_multiple_wr`

Test: `ccc_direct_multiple_wr`

Sends a sequence of multiple directed SETMWL CCCs. The first and
the last have non-matching address. The two middle ones set MWL
to different values. Verify that the target responded to correct
addresses and executed both CCCs.

##### `ccc_direct_multiple_rd`

Test: `ccc_direct_multiple_rd`

Sends SETMWL CCC. Then sends multiple directed GETMWL CCCs to
thee different addresses. Only the one for the target should
be ACK-ed with the correct MWL content.


