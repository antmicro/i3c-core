# I3C core overview

This chapter provides a high level overview of the I3C target functionalities implemented in the core.

## Configuration script

The I3C Core is configured with the [i3c_core_config](https://github.com/chipsalliance/i3c-core/blob/main/tools/i3c_config/i3c_core_config.py) Python script, which reads configurations from the `i3c_core_configs.yaml` YAML file.

The supported configurations can be found in the [i3c_core_configs.yaml](https://github.com/chipsalliance/i3c-core/blob/main/i3c_core_configs.yaml) file.

More details on the usage of the tool can be found in the [relevant README](https://github.com/chipsalliance/i3c-core/blob/main/tools/i3c_config/README.md).

## System Bus

The I3C core can connect to system buses via dedicated adapters:

* AXI4
* AHB - The AHB-Lite implementation is based on the AMBA 3 AHB-Lite Protocol Specification (IHI0033A)

## Virtual target

The I3C core implements the Secure Firmware Recovery protocol according to the Open Compute "Secure Firmware Recovery" specification rev. 1.1-rc5.
For this purpose, the I3C core exposes a "virtual" target with its own static and dynamic bus addresses.
The virtual target implementation shares most of its logic with the "main" one while retaining a distinct data path.
Certain CCC commands like `SETAASA` and `SETDASA` are implemented separately for the "main" and "virtual" targets.

## Register descriptions

Register descriptions are specified in the [RDL format](https://github.com/chipsalliance/i3c-core/tree/main/src/rdl) for I3C core CSRs:

* I3C Capability and Operational Registers (I3CBase)
* Programmable I/O (PIOControl)
* Extended Capabilities (I3C_EC)
* Device Address Table (DAT)
* Device Characteristic Table (DCT)

The RDL files generate the relevant SystemVerilog which can be found in the [src/csr/](https://github.com/chipsalliance/i3c-core/tree/main/src/csr) directory.
The auto-generated descriptions are included in the [Register descriptions](registers.md) chapter.

## Target Interface Queues

There are also target interface queues via Target Transaction Interface (TTI):

* RX - read descriptor & data queues
* TX - write descriptor & data queues
* IBI - IBI combined descriptor + data queue

## Target recovery interface

Several functionalities related to the recovery interface have been implemented for Caliptra:

* Recovery mode enable control via a CSR field
* Hardware recovery packet handling (private read/write)
* Hardware PEC checksum calculation and checking
* Access to Recovery CSRs from the I3C side
* Status signaling via output pins:

  * `recovery_payload_available_o`
  * `recovery_image_activated_o`

## Private reads and writes

* The core handles I3C private reads and writes

  * This functionality passes Avery test suite tests
  * Private writes push data to the TTI RX Queue, accessible from the AXI bus, allowing the CPU to read the data

    * The number of received bytes is written into the TTI RX descriptor Queue
    * The software is supposed to first read the descriptor data, and then the number of bytes defined by the descriptor for the TTI RX Queue

  * Private reads send data on I3C lines from TTI TX Queue

    * The software has to write the TTI TX Queue prior to a I3C private read transaction
    * The TTI TX descriptor is used similarly to set the max number of bytes to be sent in the next private read transaction

## In-Band Interrupts (IBI)

The core is capable of raising In-Band Interrupts.
IBIs are controlled using descriptors written to a dedicated IBI queue by software.
Optional IBI data immediately follows the descriptor in the same queue.

The core watches the IBI queue for a descriptor write.
Once a descriptor is written, the core peeks it and waits until the defined count of data words is written to the queue.
Finally, the core outputs the Mandatory Data Byte (MDB) and the data as 8-bit words.

## I3C Common Command Codes (CCC)

The I3C core supports all CCCs required by the I3C Basic spec, please see "Table 16 I3C Common Command Codes" for a full reference.

All CCCs are exercised with Cocotb tests.

### Broadcast CCCs

The following Broadcast CCCs are currently supported by the core (all required Broadcast CCCs as per the errata, and one optional Broadcast CCC):

* ENEC (R) - Enable Events Command
* DISEC (R) - Disable Events Command
* SETMWL (R) - Set Max Write Length
* SETMRL (R) - Set Max Read Length
* SETAASA (O) - Set All Addresses to Static Adresses
* RSTACT (R) - Target Reset Action

### Direct CCCs

The following Direct CCCs are currently supported by the core (all required Direct CCCs, plus several optional/conditional ones):

* ENEC (R) - Enable Events Command
* DISEC (R) - Disable Events Command
* RSTDAA (R) - Direct Reset Dynamic Address Assignment - this direct CCC is deprecated, the core NACKs this command as per the spec
* SETDASA (O) - Set Dynamic Address from Static Address
* SETMWL (R) - Set Max Write Length
* SETMRL (R) - Set Max Read Length
* GETMWL (R) - Get Max Write Length
* GETMRL (R) - Set Max Read Length
* GETPID (C) - Get Provisioned ID
* GETBCR (C) - Get Bus Characteristics Register
* GETDCR (C) - Get Device Characteristics Register
* GETSTATUS (R) - Get Device Status
* RSTACT (R) - Target Reset Action


## Other features

* Target Reset Pattern is detected and causes assertion of output pins, based on the action selected with RSTACT:

  * `peripheral_reset_o`
  * `escalated_reset_o`

* The core correctly detects HDR-Exit Pattern
