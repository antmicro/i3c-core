from munch import Munch

reg_map = Munch.fromDict({
    "base_addr": 0,
    "I3CBASE": {
        "start_addr": 0,
        "HCI_VERSION": {
            "base_addr": 0,
            "offset": 0,
            "VERSION": {
                "low": 0,
                "mask": 4294967295,
                "reset": 288,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "HC_CONTROL": {
            "base_addr": 4,
            "offset": 4,
            "IBA_INCLUDE": {
                "low": 0,
                "mask": 1,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "AUTOCMD_DATA_RPT": {
                "low": 3,
                "mask": 8,
                "reset": 0,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "DATA_BYTE_ORDER_MODE": {
                "low": 4,
                "mask": 16,
                "reset": 0,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "MODE_SELECTOR": {
                "low": 6,
                "mask": 64,
                "reset": 1,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "I2C_DEV_PRESENT": {
                "low": 7,
                "mask": 128,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "HOT_JOIN_CTRL": {
                "low": 8,
                "mask": 256,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "HALT_ON_CMD_SEQ_TIMEOUT": {
                "low": 12,
                "mask": 4096,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "ABORT": {
                "low": 29,
                "mask": 536870912,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "RESUME": {
                "low": 30,
                "mask": 1073741824,
                "reset": 0,
                "sw": "rw",
                "hw": "rw",
                "woclr": 1,
                "rclr": 0,
                "hwclr": 0
            },
            "BUS_ENABLE": {
                "low": 31,
                "mask": 2147483648,
                "reset": 0,
                "sw": "rw",
                "hw": "rw",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "CONTROLLER_DEVICE_ADDR": {
            "base_addr": 8,
            "offset": 8,
            "DYNAMIC_ADDR": {
                "low": 16,
                "mask": 8323072,
                "reset": 0,
                "sw": "rw",
                "hw": "rw",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "DYNAMIC_ADDR_VALID": {
                "low": 31,
                "mask": 2147483648,
                "reset": 0,
                "sw": "rw",
                "hw": "rw",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "HC_CAPABILITIES": {
            "base_addr": 12,
            "offset": 12,
            "COMBO_COMMAND": {
                "low": 2,
                "mask": 4,
                "reset": 0,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "AUTO_COMMAND": {
                "low": 3,
                "mask": 8,
                "reset": 0,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "STANDBY_CR_CAP": {
                "low": 5,
                "mask": 32,
                "reset": 0,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "HDR_DDR_EN": {
                "low": 6,
                "mask": 64,
                "reset": 0,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "HDR_TS_EN": {
                "low": 7,
                "mask": 128,
                "reset": 0,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "CMD_CCC_DEFBYTE": {
                "low": 10,
                "mask": 1024,
                "reset": 1,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "IBI_DATA_ABORT_EN": {
                "low": 11,
                "mask": 2048,
                "reset": 0,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "IBI_CREDIT_COUNT_EN": {
                "low": 12,
                "mask": 4096,
                "reset": 0,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "SCHEDULED_COMMANDS_EN": {
                "low": 13,
                "mask": 8192,
                "reset": 0,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "CMD_SIZE": {
                "low": 20,
                "mask": 3145728,
                "reset": 0,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "SG_CAPABILITY_CR_EN": {
                "low": 28,
                "mask": 268435456,
                "reset": 0,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "SG_CAPABILITY_IBI_EN": {
                "low": 29,
                "mask": 536870912,
                "reset": 0,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "SG_CAPABILITY_DC_EN": {
                "low": 30,
                "mask": 1073741824,
                "reset": 0,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "RESET_CONTROL": {
            "base_addr": 16,
            "offset": 16,
            "SOFT_RST": {
                "low": 0,
                "mask": 1,
                "reset": 0,
                "sw": "rw",
                "hw": "rw",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "CMD_QUEUE_RST": {
                "low": 1,
                "mask": 2,
                "reset": 0,
                "sw": "rw",
                "hw": "rw",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "RESP_QUEUE_RST": {
                "low": 2,
                "mask": 4,
                "reset": 0,
                "sw": "rw",
                "hw": "rw",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "TX_FIFO_RST": {
                "low": 3,
                "mask": 8,
                "reset": 0,
                "sw": "rw",
                "hw": "rw",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "RX_FIFO_RST": {
                "low": 4,
                "mask": 16,
                "reset": 0,
                "sw": "rw",
                "hw": "rw",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "IBI_QUEUE_RST": {
                "low": 5,
                "mask": 32,
                "reset": 0,
                "sw": "rw",
                "hw": "rw",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "PRESENT_STATE": {
            "base_addr": 20,
            "offset": 20,
            "AC_CURRENT_OWN": {
                "low": 2,
                "mask": 4,
                "reset": 0,
                "sw": "r",
                "hw": "w",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "INTR_STATUS": {
            "base_addr": 32,
            "offset": 32,
            "HC_INTERNAL_ERR_STAT": {
                "low": 10,
                "mask": 1024,
                "reset": 0,
                "sw": "rw",
                "hw": "w",
                "woclr": 1,
                "rclr": 0,
                "hwclr": 0
            },
            "HC_SEQ_CANCEL_STAT": {
                "low": 11,
                "mask": 2048,
                "reset": 0,
                "sw": "rw",
                "hw": "w",
                "woclr": 1,
                "rclr": 0,
                "hwclr": 0
            },
            "HC_WARN_CMD_SEQ_STALL_STAT": {
                "low": 12,
                "mask": 4096,
                "reset": 0,
                "sw": "rw",
                "hw": "w",
                "woclr": 1,
                "rclr": 0,
                "hwclr": 0
            },
            "HC_ERR_CMD_SEQ_TIMEOUT_STAT": {
                "low": 13,
                "mask": 8192,
                "reset": 0,
                "sw": "rw",
                "hw": "w",
                "woclr": 1,
                "rclr": 0,
                "hwclr": 0
            },
            "SCHED_CMD_MISSED_TICK_STAT": {
                "low": 14,
                "mask": 16384,
                "reset": 0,
                "sw": "rw",
                "hw": "w",
                "woclr": 1,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "INTR_STATUS_ENABLE": {
            "base_addr": 36,
            "offset": 36,
            "HC_INTERNAL_ERR_STAT_EN": {
                "low": 10,
                "mask": 1024,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "HC_SEQ_CANCEL_STAT_EN": {
                "low": 11,
                "mask": 2048,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "HC_WARN_CMD_SEQ_STALL_STAT_EN": {
                "low": 12,
                "mask": 4096,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "HC_ERR_CMD_SEQ_TIMEOUT_STAT_EN": {
                "low": 13,
                "mask": 8192,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "SCHED_CMD_MISSED_TICK_STAT_EN": {
                "low": 14,
                "mask": 16384,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "INTR_SIGNAL_ENABLE": {
            "base_addr": 40,
            "offset": 40,
            "HC_INTERNAL_ERR_SIGNAL_EN": {
                "low": 10,
                "mask": 1024,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "HC_SEQ_CANCEL_SIGNAL_EN": {
                "low": 11,
                "mask": 2048,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "HC_WARN_CMD_SEQ_STALL_SIGNAL_EN": {
                "low": 12,
                "mask": 4096,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "HC_ERR_CMD_SEQ_TIMEOUT_SIGNAL_EN": {
                "low": 13,
                "mask": 8192,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "SCHED_CMD_MISSED_TICK_SIGNAL_EN": {
                "low": 14,
                "mask": 16384,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "INTR_FORCE": {
            "base_addr": 44,
            "offset": 44,
            "HC_INTERNAL_ERR_FORCE": {
                "low": 10,
                "mask": 1024,
                "reset": 0,
                "sw": "w",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "HC_SEQ_CANCEL_FORCE": {
                "low": 11,
                "mask": 2048,
                "reset": 0,
                "sw": "w",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "HC_WARN_CMD_SEQ_STALL_FORCE": {
                "low": 12,
                "mask": 4096,
                "reset": 0,
                "sw": "w",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "HC_ERR_CMD_SEQ_TIMEOUT_FORCE": {
                "low": 13,
                "mask": 8192,
                "reset": 0,
                "sw": "w",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "SCHED_CMD_MISSED_TICK_FORCE": {
                "low": 14,
                "mask": 16384,
                "reset": 0,
                "sw": "w",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "DAT_SECTION_OFFSET": {
            "base_addr": 48,
            "offset": 48,
            "TABLE_OFFSET": {
                "low": 0,
                "mask": 4095,
                "reset": 1024,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "TABLE_SIZE": {
                "low": 12,
                "mask": 520192,
                "reset": 127,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "ENTRY_SIZE": {
                "low": 28,
                "mask": 4026531840,
                "reset": 0,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "DCT_SECTION_OFFSET": {
            "base_addr": 52,
            "offset": 52,
            "TABLE_OFFSET": {
                "low": 0,
                "mask": 4095,
                "reset": 2048,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "TABLE_SIZE": {
                "low": 12,
                "mask": 520192,
                "reset": 127,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "TABLE_INDEX": {
                "low": 19,
                "mask": 16252928,
                "reset": 0,
                "sw": "rw",
                "hw": "rw",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "ENTRY_SIZE": {
                "low": 28,
                "mask": 4026531840,
                "reset": 0,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "RING_HEADERS_SECTION_OFFSET": {
            "base_addr": 56,
            "offset": 56,
            "SECTION_OFFSET": {
                "low": 0,
                "mask": 65535,
                "reset": 0,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "PIO_SECTION_OFFSET": {
            "base_addr": 60,
            "offset": 60,
            "SECTION_OFFSET": {
                "low": 0,
                "mask": 65535,
                "reset": 128,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "EXT_CAPS_SECTION_OFFSET": {
            "base_addr": 64,
            "offset": 64,
            "SECTION_OFFSET": {
                "low": 0,
                "mask": 65535,
                "reset": 256,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "INT_CTRL_CMDS_EN": {
            "base_addr": 76,
            "offset": 76,
            "ICC_SUPPORT": {
                "low": 0,
                "mask": 1,
                "reset": 1,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "MIPI_CMDS_SUPPORTED": {
                "low": 1,
                "mask": 65534,
                "reset": 53,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "IBI_NOTIFY_CTRL": {
            "base_addr": 88,
            "offset": 88,
            "NOTIFY_HJ_REJECTED": {
                "low": 0,
                "mask": 1,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "NOTIFY_CRR_REJECTED": {
                "low": 1,
                "mask": 2,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "NOTIFY_IBI_REJECTED": {
                "low": 3,
                "mask": 8,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "IBI_DATA_ABORT_CTRL": {
            "base_addr": 92,
            "offset": 92,
            "MATCH_IBI_ID": {
                "low": 8,
                "mask": 65280,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "AFTER_N_CHUNKS": {
                "low": 16,
                "mask": 196608,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "MATCH_STATUS_TYPE": {
                "low": 18,
                "mask": 1835008,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "IBI_DATA_ABORT_MON": {
                "low": 31,
                "mask": 2147483648,
                "reset": 0,
                "sw": "rw",
                "hw": "rw",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "DEV_CTX_BASE_LO": {
            "base_addr": 96,
            "offset": 96,
            "BASE_LO": {
                "low": 0,
                "mask": 1,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "DEV_CTX_BASE_HI": {
            "base_addr": 100,
            "offset": 100,
            "BASE_HI": {
                "low": 0,
                "mask": 1,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "DEV_CTX_SG": {
            "base_addr": 104,
            "offset": 104,
            "LIST_SIZE": {
                "low": 0,
                "mask": 65535,
                "reset": 0,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "BLP": {
                "low": 31,
                "mask": 2147483648,
                "reset": 0,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        }
    },
    "PIOCONTROL": {
        "start_addr": 128,
        "COMMAND_PORT": {
            "base_addr": 128,
            "offset": 128,
            "COMMAND_DATA": {
                "low": 0,
                "mask": 4294967295,
                "sw": "w",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "RESPONSE_PORT": {
            "base_addr": 132,
            "offset": 132,
            "RESPONSE_DATA": {
                "low": 0,
                "mask": 4294967295,
                "sw": "r",
                "hw": "w",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "TX_DATA_PORT": {
            "base_addr": 136,
            "offset": 136,
            "TX_DATA": {
                "low": 0,
                "mask": 4294967295,
                "sw": "w",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "RX_DATA_PORT": {
            "base_addr": 136,
            "offset": 136,
            "RX_DATA": {
                "low": 0,
                "mask": 4294967295,
                "sw": "r",
                "hw": "w",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "IBI_PORT": {
            "base_addr": 140,
            "offset": 140,
            "IBI_DATA": {
                "low": 0,
                "mask": 4294967295,
                "sw": "r",
                "hw": "w",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "QUEUE_THLD_CTRL": {
            "base_addr": 144,
            "offset": 144,
            "CMD_EMPTY_BUF_THLD": {
                "low": 0,
                "mask": 255,
                "reset": 1,
                "sw": "rw",
                "hw": "rw",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "RESP_BUF_THLD": {
                "low": 8,
                "mask": 65280,
                "reset": 1,
                "sw": "rw",
                "hw": "rw",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "IBI_DATA_SEGMENT_SIZE": {
                "low": 16,
                "mask": 16711680,
                "reset": 1,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "IBI_STATUS_THLD": {
                "low": 24,
                "mask": 4278190080,
                "reset": 1,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "DATA_BUFFER_THLD_CTRL": {
            "base_addr": 148,
            "offset": 148,
            "TX_BUF_THLD": {
                "low": 0,
                "mask": 7,
                "reset": 1,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "RX_BUF_THLD": {
                "low": 8,
                "mask": 1792,
                "reset": 1,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "TX_START_THLD": {
                "low": 16,
                "mask": 458752,
                "reset": 1,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "RX_START_THLD": {
                "low": 24,
                "mask": 117440512,
                "reset": 1,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "QUEUE_SIZE": {
            "base_addr": 152,
            "offset": 152,
            "CR_QUEUE_SIZE": {
                "low": 0,
                "mask": 255,
                "reset": 64,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "IBI_STATUS_SIZE": {
                "low": 8,
                "mask": 65280,
                "reset": 64,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "RX_DATA_BUFFER_SIZE": {
                "low": 16,
                "mask": 16711680,
                "reset": 5,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "TX_DATA_BUFFER_SIZE": {
                "low": 24,
                "mask": 4278190080,
                "reset": 5,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "ALT_QUEUE_SIZE": {
            "base_addr": 156,
            "offset": 156,
            "ALT_RESP_QUEUE_SIZE": {
                "low": 0,
                "mask": 255,
                "reset": 64,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "ALT_RESP_QUEUE_EN": {
                "low": 24,
                "mask": 16777216,
                "reset": 0,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "EXT_IBI_QUEUE_EN": {
                "low": 28,
                "mask": 268435456,
                "reset": 0,
                "sw": "r",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "PIO_INTR_STATUS": {
            "base_addr": 160,
            "offset": 160,
            "TX_THLD_STAT": {
                "low": 0,
                "mask": 1,
                "reset": 0,
                "sw": "r",
                "hw": "w",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "RX_THLD_STAT": {
                "low": 1,
                "mask": 2,
                "reset": 0,
                "sw": "r",
                "hw": "w",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "IBI_STATUS_THLD_STAT": {
                "low": 2,
                "mask": 4,
                "reset": 0,
                "sw": "r",
                "hw": "w",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "CMD_QUEUE_READY_STAT": {
                "low": 3,
                "mask": 8,
                "reset": 0,
                "sw": "r",
                "hw": "w",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "RESP_READY_STAT": {
                "low": 4,
                "mask": 16,
                "reset": 0,
                "sw": "r",
                "hw": "w",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "TRANSFER_ABORT_STAT": {
                "low": 5,
                "mask": 32,
                "reset": 0,
                "sw": "rw",
                "hw": "w",
                "woclr": 1,
                "rclr": 0,
                "hwclr": 0
            },
            "TRANSFER_ERR_STAT": {
                "low": 9,
                "mask": 512,
                "reset": 0,
                "sw": "rw",
                "hw": "w",
                "woclr": 1,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "PIO_INTR_STATUS_ENABLE": {
            "base_addr": 164,
            "offset": 164,
            "TX_THLD_STAT_EN": {
                "low": 0,
                "mask": 1,
                "reset": 0,
                "sw": "rw",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "RX_THLD_STAT_EN": {
                "low": 1,
                "mask": 2,
                "reset": 0,
                "sw": "rw",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "IBI_STATUS_THLD_STAT_EN": {
                "low": 2,
                "mask": 4,
                "reset": 0,
                "sw": "rw",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "CMD_QUEUE_READY_STAT_EN": {
                "low": 3,
                "mask": 8,
                "reset": 0,
                "sw": "rw",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "RESP_READY_STAT_EN": {
                "low": 4,
                "mask": 16,
                "reset": 0,
                "sw": "rw",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "TRANSFER_ABORT_STAT_EN": {
                "low": 5,
                "mask": 32,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "TRANSFER_ERR_STAT_EN": {
                "low": 9,
                "mask": 512,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "PIO_INTR_SIGNAL_ENABLE": {
            "base_addr": 168,
            "offset": 168,
            "TX_THLD_SIGNAL_EN": {
                "low": 0,
                "mask": 1,
                "reset": 0,
                "sw": "rw",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "RX_THLD_SIGNAL_EN": {
                "low": 1,
                "mask": 2,
                "reset": 0,
                "sw": "rw",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "IBI_STATUS_THLD_SIGNAL_EN": {
                "low": 2,
                "mask": 4,
                "reset": 0,
                "sw": "rw",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "CMD_QUEUE_READY_SIGNAL_EN": {
                "low": 3,
                "mask": 8,
                "reset": 0,
                "sw": "rw",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "RESP_READY_SIGNAL_EN": {
                "low": 4,
                "mask": 16,
                "reset": 0,
                "sw": "rw",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "TRANSFER_ABORT_SIGNAL_EN": {
                "low": 5,
                "mask": 32,
                "reset": 0,
                "sw": "rw",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "TRANSFER_ERR_SIGNAL_EN": {
                "low": 9,
                "mask": 512,
                "reset": 0,
                "sw": "rw",
                "hw": "na",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "PIO_INTR_FORCE": {
            "base_addr": 172,
            "offset": 172,
            "TX_THLD_FORCE": {
                "low": 0,
                "mask": 1,
                "reset": 0,
                "sw": "w",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "RX_THLD_FORCE": {
                "low": 1,
                "mask": 2,
                "reset": 0,
                "sw": "w",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "IBI_THLD_FORCE": {
                "low": 2,
                "mask": 4,
                "reset": 0,
                "sw": "w",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "CMD_QUEUE_READY_FORCE": {
                "low": 3,
                "mask": 8,
                "reset": 0,
                "sw": "w",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "RESP_READY_FORCE": {
                "low": 4,
                "mask": 16,
                "reset": 0,
                "sw": "w",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "TRANSFER_ABORT_FORCE": {
                "low": 5,
                "mask": 32,
                "reset": 0,
                "sw": "w",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "TRANSFER_ERR_FORCE": {
                "low": 9,
                "mask": 512,
                "reset": 0,
                "sw": "w",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        },
        "PIO_CONTROL": {
            "base_addr": 176,
            "offset": 176,
            "ENABLE": {
                "low": 0,
                "mask": 1,
                "reset": 1,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "RS": {
                "low": 1,
                "mask": 2,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "ABORT": {
                "low": 2,
                "mask": 4,
                "reset": 0,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        }
    },
    "I3C_EC": {
        "start_addr": 256,
        "STDBYCTRLMODE": {
            "start_addr": 384,
            "EXTCAP_HEADER": {
                "base_addr": 384,
                "offset": 384,
                "CAP_ID": {
                    "low": 0,
                    "mask": 255,
                    "reset": 18,
                    "sw": "r",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "CAP_LENGTH": {
                    "low": 8,
                    "mask": 16776960,
                    "reset": 32,
                    "sw": "r",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "STBY_CR_CONTROL": {
                "base_addr": 388,
                "offset": 388,
                "PENDING_RX_NACK": {
                    "low": 0,
                    "mask": 1,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "HANDOFF_DELAY_NACK": {
                    "low": 1,
                    "mask": 2,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "ACR_FSM_OP_SELECT": {
                    "low": 2,
                    "mask": 4,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "PRIME_ACCEPT_GETACCCR": {
                    "low": 3,
                    "mask": 8,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "HANDOFF_DEEP_SLEEP": {
                    "low": 4,
                    "mask": 16,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 1
                },
                "CR_REQUEST_SEND": {
                    "low": 5,
                    "mask": 32,
                    "reset": 0,
                    "sw": "w",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "BAST_CCC_IBI_RING": {
                    "low": 8,
                    "mask": 1792,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "TARGET_XACT_ENABLE": {
                    "low": 12,
                    "mask": 4096,
                    "reset": 1,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "DAA_SETAASA_ENABLE": {
                    "low": 13,
                    "mask": 8192,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "DAA_SETDASA_ENABLE": {
                    "low": 14,
                    "mask": 16384,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "DAA_ENTDAA_ENABLE": {
                    "low": 15,
                    "mask": 32768,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "RSTACT_DEFBYTE_02": {
                    "low": 20,
                    "mask": 1048576,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "STBY_CR_ENABLE_INIT": {
                    "low": 30,
                    "mask": 3221225472,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "STBY_CR_DEVICE_ADDR": {
                "base_addr": 392,
                "offset": 392,
                "STATIC_ADDR": {
                    "low": 0,
                    "mask": 127,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "STATIC_ADDR_VALID": {
                    "low": 15,
                    "mask": 32768,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "DYNAMIC_ADDR": {
                    "low": 16,
                    "mask": 8323072,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "DYNAMIC_ADDR_VALID": {
                    "low": 31,
                    "mask": 2147483648,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "STBY_CR_CAPABILITIES": {
                "base_addr": 396,
                "offset": 396,
                "SIMPLE_CRR_SUPPORT": {
                    "low": 5,
                    "mask": 32,
                    "reset": 0,
                    "sw": "r",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "TARGET_XACT_SUPPORT": {
                    "low": 12,
                    "mask": 4096,
                    "reset": 1,
                    "sw": "r",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "DAA_SETAASA_SUPPORT": {
                    "low": 13,
                    "mask": 8192,
                    "reset": 1,
                    "sw": "r",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "DAA_SETDASA_SUPPORT": {
                    "low": 14,
                    "mask": 16384,
                    "reset": 1,
                    "sw": "r",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "DAA_ENTDAA_SUPPORT": {
                    "low": 15,
                    "mask": 32768,
                    "reset": 0,
                    "sw": "r",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "STBY_CR_VIRTUAL_DEVICE_CHAR": {
                "base_addr": 400,
                "offset": 400,
                "PID_HI": {
                    "low": 1,
                    "mask": 65534,
                    "reset": 32767,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "DCR": {
                    "low": 16,
                    "mask": 16711680,
                    "reset": 189,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "BCR_VAR": {
                    "low": 24,
                    "mask": 520093696,
                    "reset": 16,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "BCR_FIXED": {
                    "low": 29,
                    "mask": 3758096384,
                    "reset": 1,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "STBY_CR_STATUS": {
                "base_addr": 404,
                "offset": 404,
                "AC_CURRENT_OWN": {
                    "low": 2,
                    "mask": 4,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "SIMPLE_CRR_STATUS": {
                    "low": 5,
                    "mask": 224,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "HJ_REQ_STATUS": {
                    "low": 8,
                    "mask": 256,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "STBY_CR_DEVICE_CHAR": {
                "base_addr": 408,
                "offset": 408,
                "PID_HI": {
                    "low": 1,
                    "mask": 65534,
                    "reset": 32767,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "DCR": {
                    "low": 16,
                    "mask": 16711680,
                    "reset": 189,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "BCR_VAR": {
                    "low": 24,
                    "mask": 520093696,
                    "reset": 22,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "BCR_FIXED": {
                    "low": 29,
                    "mask": 3758096384,
                    "reset": 1,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "STBY_CR_DEVICE_PID_LO": {
                "base_addr": 412,
                "offset": 412,
                "PID_LO": {
                    "low": 0,
                    "mask": 4294967295,
                    "reset": 5898405,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "STBY_CR_INTR_STATUS": {
                "base_addr": 416,
                "offset": 416,
                "ACR_HANDOFF_OK_REMAIN_STAT": {
                    "low": 0,
                    "mask": 1,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "ACR_HANDOFF_OK_PRIMED_STAT": {
                    "low": 1,
                    "mask": 2,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "ACR_HANDOFF_ERR_FAIL_STAT": {
                    "low": 2,
                    "mask": 4,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "ACR_HANDOFF_ERR_M3_STAT": {
                    "low": 3,
                    "mask": 8,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "CRR_RESPONSE_STAT": {
                    "low": 10,
                    "mask": 1024,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "STBY_CR_DYN_ADDR_STAT": {
                    "low": 11,
                    "mask": 2048,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "STBY_CR_ACCEPT_NACKED_STAT": {
                    "low": 12,
                    "mask": 4096,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "STBY_CR_ACCEPT_OK_STAT": {
                    "low": 13,
                    "mask": 8192,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "STBY_CR_ACCEPT_ERR_STAT": {
                    "low": 14,
                    "mask": 16384,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "STBY_CR_OP_RSTACT_STAT": {
                    "low": 16,
                    "mask": 65536,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "CCC_PARAM_MODIFIED_STAT": {
                    "low": 17,
                    "mask": 131072,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "CCC_UNHANDLED_NACK_STAT": {
                    "low": 18,
                    "mask": 262144,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "CCC_FATAL_RSTDAA_ERR_STAT": {
                    "low": 19,
                    "mask": 524288,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "STBY_CR_VIRTUAL_DEVICE_PID_LO": {
                "base_addr": 420,
                "offset": 420,
                "PID_LO": {
                    "low": 0,
                    "mask": 4294967295,
                    "reset": 5902501,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "STBY_CR_INTR_SIGNAL_ENABLE": {
                "base_addr": 424,
                "offset": 424,
                "ACR_HANDOFF_OK_REMAIN_SIGNAL_EN": {
                    "low": 0,
                    "mask": 1,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "ACR_HANDOFF_OK_PRIMED_SIGNAL_EN": {
                    "low": 1,
                    "mask": 2,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "ACR_HANDOFF_ERR_FAIL_SIGNAL_EN": {
                    "low": 2,
                    "mask": 4,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "ACR_HANDOFF_ERR_M3_SIGNAL_EN": {
                    "low": 3,
                    "mask": 8,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "CRR_RESPONSE_SIGNAL_EN": {
                    "low": 10,
                    "mask": 1024,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "STBY_CR_DYN_ADDR_SIGNAL_EN": {
                    "low": 11,
                    "mask": 2048,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "STBY_CR_ACCEPT_NACKED_SIGNAL_EN": {
                    "low": 12,
                    "mask": 4096,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "STBY_CR_ACCEPT_OK_SIGNAL_EN": {
                    "low": 13,
                    "mask": 8192,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "STBY_CR_ACCEPT_ERR_SIGNAL_EN": {
                    "low": 14,
                    "mask": 16384,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "STBY_CR_OP_RSTACT_SIGNAL_EN": {
                    "low": 16,
                    "mask": 65536,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "CCC_PARAM_MODIFIED_SIGNAL_EN": {
                    "low": 17,
                    "mask": 131072,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "CCC_UNHANDLED_NACK_SIGNAL_EN": {
                    "low": 18,
                    "mask": 262144,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "CCC_FATAL_RSTDAA_ERR_SIGNAL_EN": {
                    "low": 19,
                    "mask": 524288,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "STBY_CR_INTR_FORCE": {
                "base_addr": 428,
                "offset": 428,
                "CRR_RESPONSE_FORCE": {
                    "low": 10,
                    "mask": 1024,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "STBY_CR_DYN_ADDR_FORCE": {
                    "low": 11,
                    "mask": 2048,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "STBY_CR_ACCEPT_NACKED_FORCE": {
                    "low": 12,
                    "mask": 4096,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "STBY_CR_ACCEPT_OK_FORCE": {
                    "low": 13,
                    "mask": 8192,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "STBY_CR_ACCEPT_ERR_FORCE": {
                    "low": 14,
                    "mask": 16384,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "STBY_CR_OP_RSTACT_FORCE": {
                    "low": 16,
                    "mask": 65536,
                    "reset": 0,
                    "sw": "w",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "CCC_PARAM_MODIFIED_FORCE": {
                    "low": 17,
                    "mask": 131072,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "CCC_UNHANDLED_NACK_FORCE": {
                    "low": 18,
                    "mask": 262144,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "CCC_FATAL_RSTDAA_ERR_FORCE": {
                    "low": 19,
                    "mask": 524288,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "STBY_CR_CCC_CONFIG_GETCAPS": {
                "base_addr": 432,
                "offset": 432,
                "F2_CRCAP1_BUS_CONFIG": {
                    "low": 0,
                    "mask": 7,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "F2_CRCAP2_DEV_INTERACT": {
                    "low": 8,
                    "mask": 3840,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "STBY_CR_CCC_CONFIG_RSTACT_PARAMS": {
                "base_addr": 436,
                "offset": 436,
                "RST_ACTION": {
                    "low": 0,
                    "mask": 255,
                    "reset": 1,
                    "sw": "r",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "RESET_TIME_PERIPHERAL": {
                    "low": 8,
                    "mask": 65280,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "RESET_TIME_TARGET": {
                    "low": 16,
                    "mask": 16711680,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "RESET_DYNAMIC_ADDR": {
                    "low": 31,
                    "mask": 2147483648,
                    "reset": 1,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "STBY_CR_VIRT_DEVICE_ADDR": {
                "base_addr": 440,
                "offset": 440,
                "VIRT_STATIC_ADDR": {
                    "low": 0,
                    "mask": 127,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "VIRT_STATIC_ADDR_VALID": {
                    "low": 15,
                    "mask": 32768,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "VIRT_DYNAMIC_ADDR": {
                    "low": 16,
                    "mask": 8323072,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "VIRT_DYNAMIC_ADDR_VALID": {
                    "low": 31,
                    "mask": 2147483648,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "STBY_CR_MWL": {
                "base_addr": 444,
                "offset": 444,
                "MWL": {
                    "low": 0,
                    "mask": 65535,
                    "reset": 256,
                    "sw": "r",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "STBY_CR_MRL": {
                "base_addr": 448,
                "offset": 448,
                "MRL": {
                    "low": 0,
                    "mask": 65535,
                    "reset": 256,
                    "sw": "r",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "IBIL": {
                    "low": 16,
                    "mask": 16711680,
                    "reset": 16,
                    "sw": "r",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            }
        },
        "SOCMGMTIF": {
            "start_addr": 768,
            "EXTCAP_HEADER": {
                "base_addr": 768,
                "offset": 768,
                "CAP_ID": {
                    "low": 0,
                    "mask": 255,
                    "reset": 193,
                    "sw": "r",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "CAP_LENGTH": {
                    "low": 8,
                    "mask": 16776960,
                    "reset": 26,
                    "sw": "r",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "SOC_MGMT_CONTROL": {
                "base_addr": 772,
                "offset": 772,
                "PLACEHOLDER": {
                    "low": 0,
                    "mask": 4294967295,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "SOC_MGMT_STATUS": {
                "base_addr": 776,
                "offset": 776,
                "PLACEHOLDER": {
                    "low": 0,
                    "mask": 4294967295,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "REC_INTF_CFG": {
                "base_addr": 780,
                "offset": 780,
                "REC_INTF_BYPASS": {
                    "low": 0,
                    "mask": 1,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "REC_PAYLOAD_DONE": {
                    "low": 1,
                    "mask": 2,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "REC_INTF_REG_W1C_ACCESS": {
                "base_addr": 784,
                "offset": 784,
                "DEVICE_RESET_CTRL": {
                    "low": 0,
                    "mask": 255,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "RECOVERY_CTRL_ACTIVATE_REC_IMG": {
                    "low": 8,
                    "mask": 65280,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "INDIRECT_FIFO_CTRL_RESET": {
                    "low": 16,
                    "mask": 16711680,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "SOC_MGMT_RSVD_2": {
                "base_addr": 788,
                "offset": 788,
                "PLACEHOLDER": {
                    "low": 0,
                    "mask": 4294967295,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "SOC_MGMT_RSVD_3": {
                "base_addr": 792,
                "offset": 792,
                "PLACEHOLDER": {
                    "low": 0,
                    "mask": 4294967295,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "SOC_PAD_CONF": {
                "base_addr": 796,
                "offset": 796,
                "INPUT_ENABLE": {
                    "low": 0,
                    "mask": 1,
                    "reset": 1,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "SCHMITT_EN": {
                    "low": 1,
                    "mask": 2,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "KEEPER_EN": {
                    "low": 2,
                    "mask": 4,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "PULL_DIR": {
                    "low": 3,
                    "mask": 8,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "PULL_EN": {
                    "low": 4,
                    "mask": 16,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "IO_INVERSION": {
                    "low": 5,
                    "mask": 32,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "OD_EN": {
                    "low": 6,
                    "mask": 64,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "VIRTUAL_OD_EN": {
                    "low": 7,
                    "mask": 128,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "PAD_TYPE": {
                    "low": 24,
                    "mask": 4278190080,
                    "reset": 1,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "SOC_PAD_ATTR": {
                "base_addr": 800,
                "offset": 800,
                "DRIVE_SLEW_RATE": {
                    "low": 8,
                    "mask": 65280,
                    "reset": 15,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "DRIVE_STRENGTH": {
                    "low": 24,
                    "mask": 4278190080,
                    "reset": 15,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "SOC_MGMT_FEATURE_2": {
                "base_addr": 804,
                "offset": 804,
                "PLACEHOLDER": {
                    "low": 0,
                    "mask": 4294967295,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "SOC_MGMT_FEATURE_3": {
                "base_addr": 808,
                "offset": 808,
                "PLACEHOLDER": {
                    "low": 0,
                    "mask": 4294967295,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "T_R_REG": {
                "base_addr": 812,
                "offset": 812,
                "T_R": {
                    "low": 0,
                    "mask": 1048575,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "T_F_REG": {
                "base_addr": 816,
                "offset": 816,
                "T_F": {
                    "low": 0,
                    "mask": 1048575,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "T_SU_DAT_REG": {
                "base_addr": 820,
                "offset": 820,
                "T_SU_DAT": {
                    "low": 0,
                    "mask": 1048575,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "T_SU_DAT_I2C_REG": {
                "base_addr": 824,
                "offset": 824,
                "T_SU_DAT_I2C": {
                    "low": 0,
                    "mask": 1048575,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "T_HD_DAT_REG": {
                "base_addr": 828,
                "offset": 828,
                "T_HD_DAT": {
                    "low": 0,
                    "mask": 1048575,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "T_HIGH_REG": {
                "base_addr": 832,
                "offset": 832,
                "T_HIGH": {
                    "low": 0,
                    "mask": 1048575,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "T_HIGH_OD_REG": {
                "base_addr": 836,
                "offset": 836,
                "T_HIGH_OD": {
                    "low": 0,
                    "mask": 1048575,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "T_HIGH_INIT_OD_REG": {
                "base_addr": 840,
                "offset": 840,
                "T_HIGH_INIT_OD": {
                    "low": 0,
                    "mask": 1048575,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "T_HIGH_I2C_REG": {
                "base_addr": 844,
                "offset": 844,
                "T_HIGH_I2C": {
                    "low": 0,
                    "mask": 1048575,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "T_LOW_REG": {
                "base_addr": 848,
                "offset": 848,
                "T_LOW": {
                    "low": 0,
                    "mask": 1048575,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "T_LOW_OD_REG": {
                "base_addr": 852,
                "offset": 852,
                "T_LOW_OD": {
                    "low": 0,
                    "mask": 1048575,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "T_LOW_I2C_REG": {
                "base_addr": 856,
                "offset": 856,
                "T_LOW_I2C": {
                    "low": 0,
                    "mask": 1048575,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "T_HD_STA_REG": {
                "base_addr": 860,
                "offset": 860,
                "T_HD_STA": {
                    "low": 0,
                    "mask": 1048575,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "T_HD_STA_I2C_REG": {
                "base_addr": 864,
                "offset": 864,
                "T_HD_STA_I2C": {
                    "low": 0,
                    "mask": 1048575,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "T_HD_RSTA_REG": {
                "base_addr": 868,
                "offset": 868,
                "T_HD_RSTA": {
                    "low": 0,
                    "mask": 1048575,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "T_SU_STA_REG": {
                "base_addr": 872,
                "offset": 872,
                "T_SU_STA": {
                    "low": 0,
                    "mask": 1048575,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "T_SU_STA_I2C_REG": {
                "base_addr": 876,
                "offset": 876,
                "T_SU_STA_I2C": {
                    "low": 0,
                    "mask": 1048575,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "T_SU_STO_REG": {
                "base_addr": 880,
                "offset": 880,
                "T_SU_STO": {
                    "low": 0,
                    "mask": 1048575,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "T_SU_STO_I2C_REG": {
                "base_addr": 884,
                "offset": 884,
                "T_SU_STO_I2C": {
                    "low": 0,
                    "mask": 1048575,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "T_DS_OD_REG": {
                "base_addr": 888,
                "offset": 888,
                "T_DS_OD": {
                    "low": 0,
                    "mask": 1048575,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "T_FREE_REG": {
                "base_addr": 892,
                "offset": 892,
                "T_FREE": {
                    "low": 0,
                    "mask": 4294967295,
                    "reset": 12,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "T_FREE_I2C_REG": {
                "base_addr": 896,
                "offset": 896,
                "T_FREE_I2C": {
                    "low": 0,
                    "mask": 4294967295,
                    "reset": 12,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "T_AVAL_REG": {
                "base_addr": 900,
                "offset": 900,
                "T_AVAL": {
                    "low": 0,
                    "mask": 4294967295,
                    "reset": 300,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "T_IDLE_REG": {
                "base_addr": 904,
                "offset": 904,
                "T_IDLE": {
                    "low": 0,
                    "mask": 4294967295,
                    "reset": 60000,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "HDR_TIMEOUT_EN_REG": {
                "base_addr": 908,
                "offset": 908,
                "HDR_TIMEOUT_EN": {
                    "low": 0,
                    "mask": 1,
                    "reset": 0,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "T_HDR_TIMEOUT_REG": {
                "base_addr": 912,
                "offset": 912,
                "T_HDR_TIMEOUT": {
                    "low": 0,
                    "mask": 1048575,
                    "reset": 60000,
                    "sw": "rw",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            }
        },
        "CTRLCFG": {
            "start_addr": 920,
            "EXTCAP_HEADER": {
                "base_addr": 920,
                "offset": 920,
                "CAP_ID": {
                    "low": 0,
                    "mask": 255,
                    "reset": 2,
                    "sw": "r",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                },
                "CAP_LENGTH": {
                    "low": 8,
                    "mask": 16776960,
                    "reset": 2,
                    "sw": "r",
                    "hw": "r",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            },
            "CONTROLLER_CONFIG": {
                "base_addr": 924,
                "offset": 924,
                "OPERATION_MODE": {
                    "low": 4,
                    "mask": 48,
                    "reset": 1,
                    "sw": "r",
                    "hw": "rw",
                    "woclr": 0,
                    "rclr": 0,
                    "hwclr": 0
                }
            }
        },
        "TERMINATION_EXTCAP_HEADER": {
            "base_addr": 928,
            "offset": 928,
            "CAP_ID": {
                "low": 0,
                "mask": 255,
                "reset": 0,
                "sw": "r",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "CAP_LENGTH": {
                "low": 8,
                "mask": 16776960,
                "reset": 1,
                "sw": "r",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        }
    },
    "DAT": {
        "start_addr": 1024,
        "DAT_MEMORY": {
            "base_addr": 1024,
            "offset": 1024,
            "STATIC_ADDRESS": {
                "low": 0,
                "mask": 127,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "IBI_PAYLOAD": {
                "low": 12,
                "mask": 4096,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "IBI_REJECT": {
                "low": 13,
                "mask": 8192,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "CRR_REJECT": {
                "low": 14,
                "mask": 16384,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "TS": {
                "low": 15,
                "mask": 32768,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "DYNAMIC_ADDRESS": {
                "low": 16,
                "mask": 16711680,
                "sw": "rw",
                "hw": "rw",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "RING_ID": {
                "low": 26,
                "mask": 469762048,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "DEV_NACK_RETRY_CNT": {
                "low": 29,
                "mask": 1610612736,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "DEVICE": {
                "low": 31,
                "mask": 2147483648,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "AUTOCMD_MASK": {
                "low": 32,
                "mask": 255,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "AUTOCMD_VALUE": {
                "low": 40,
                "mask": 65280,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "AUTOCMD_MODE": {
                "low": 48,
                "mask": 458752,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "AUTOCMD_HDR_CODE": {
                "low": 51,
                "mask": 133693440,
                "sw": "rw",
                "hw": "r",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        }
    },
    "DCT": {
        "start_addr": 2048,
        "DCT_MEMORY": {
            "base_addr": 2048,
            "offset": 2048,
            "PID_HI": {
                "low": 0,
                "mask": 4294967295,
                "sw": "r",
                "hw": "rw",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "PID_LO": {
                "low": 32,
                "mask": 65535,
                "sw": "r",
                "hw": "rw",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "DCR": {
                "low": 64,
                "mask": 255,
                "sw": "r",
                "hw": "rw",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "BCR": {
                "low": 72,
                "mask": 65280,
                "sw": "r",
                "hw": "rw",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            },
            "DYNAMIC_ADDRESS": {
                "low": 96,
                "mask": 255,
                "sw": "r",
                "hw": "rw",
                "woclr": 0,
                "rclr": 0,
                "hwclr": 0
            }
        }
    }
})
