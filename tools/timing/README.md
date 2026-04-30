# I3C Timing Generator & Validator

This module calculates and validates I3C and I2C Fast Mode timing parameters (in system clock cycles) for an I3C Controller based on the MIPI I3C Basic specification.

## Core Functions

* **`generate_timings(f_scl, f_sys, duty_cycle=0.5)`**: Calculates required CSR register cycle counts based on target SCL and system clock frequencies (in Hz). Clamps max SCL frequency to 12.9 MHz and automatically enforces physical hardware minimums.
* **`validate_timings(timings, f_sys)`**: Validates a dictionary of timing cycles against absolute I3C nanosecond constraints, logging errors for any spec violations.
* **`log_timing_configuration(timings, f_sys=None)`**: Neatly formats and logs the generated timing configuration into a table. If `f_sys` is provided, it also calculates and displays the physical time in nanoseconds for each register.

## Quick Start

```python

import logging
from timings import generate_timings, log_timing_configuration

logging.basicConfig(level=logging.INFO)

# Generate valid CSR timings for 12.5 MHz SCL and 333.33 MHz System Clock
SYS_CLK = 333.333e6
csr_timings = generate_timings(f_scl=12.5e6, f_sys=SYS_CLK)

# Print the nicely formatted table of cycle counts and physical times
log_timing_configuration(csr_timings, f_sys=SYS_CLK)

```

## Usage

The `timing.py` script can be invoked to display register values corresponding to the clock settings provided in the command arguments:

```python
usage: timings.py [-h] [--freq FREQ] [--bus_freq BUS_FREQ] [--duty_cycle DUTY_CYCLE] [--md] [--target_name TARGET_NAME]

Generate and validate I3C timings.

options:
  -h, --help            show this help message and exit
  --freq FREQ           System clock frequency in Hz (Default: 200.0e6)
  --bus_freq BUS_FREQ   Target I3C bus frequency in Hz (Default: 12.5e6)
  --duty_cycle DUTY_CYCLE
                        Target duty cycle (Default: 0.5)
  --md                  Output the results as a MyST Markdown table (suppresses standard logging)
  --target_name TARGET_NAME
                        Name of the configuration for the Markdown header (e.g., 'FPGA', 'ASIC')

```
