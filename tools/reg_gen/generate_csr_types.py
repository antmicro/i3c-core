import os
import sys
from pathlib import Path

# To add a new register you have to define a mapping here
TRAITS_MAP = {
    "__out_t": "hwif_out_t",
    "__in_t": "hwif_in_t",
    "__I3CBase__out_t": "base_out_t",
    "__I3CBase__in_t": "base_in_t",
    "__PIOControl__out_t": "pio_out_t",
    "__PIOControl__in_t": "pio_in_t",
    "__I3C_EC__out_t": "ec_out_t",
    "__I3C_EC__in_t": "ec_in_t",
    "__I3C_EC__SecFwRecoveryIf__out_t" : "secfwrecoveryif_out_t",
    "__I3C_EC__SecFwRecoveryIf__in_t" : "secfwrecoveryif_in_t",
    "__I3C_EC__StdbyCtrlMode__out_t": "stdby_out_t",
    "__I3C_EC__StdbyCtrlMode__in_t": "stdby_in_t",
    "__I3C_EC__SoCMgmtIf__out_t": "socmgmt_out_t",
    "__I3C_EC__SoCMgmtIf__in_t": "socmgmt_in_t",
    "__I3C_EC__CtrlCfg__out_t": "ctrlcfg_out_t",
    "__I3C_EC__CtrlCfg__in_t": "ctrlcfg_in_t",
    "__I3C_EC__TTI__in_t" : "tti_in_t",
    "__I3C_EC__TTI__out_t" : "tti_out_t",
    "__DAT__out_t": "dat_out_t",
    "__DAT__in_t": "dat_in_t",
    "__DCT__out_t": "dct_out_t",
    "__DCT__in_t": "dct_in_t",
}

CONFIGS = [
    {
        "class_name": "controller_and_target_csr_t",
        "pkg_file": "controller_and_target_I3CCSR_pkg.sv",
        "pkg_name": "controller_and_target_I3CCSR_pkg",
        "prefix": "controller_and_target_I3CCSR"
    },
    {
        "class_name": "target_csr_t",
        "pkg_file": "target_I3CCSR_pkg.sv",
        "pkg_name": "target_I3CCSR_pkg",
        "prefix": "target_I3CCSR"
    },
    {
        "class_name": "controller_csr_t",
        "pkg_file": "controller_I3CCSR_pkg.sv",
        "pkg_name": "controller_I3CCSR_pkg",
        "prefix": "controller_I3CCSR"
    }
]

def generate_csr_types(source_dir, output_file):
    out_lines = [
        "// ============================================================================",
        "// AUTO-GENERATED I3C CSR Traits Definitions",
        "// Generated from PeakRDL package files to abstract configuration types.",
        "// ============================================================================",
        "",
        "`ifndef I3C_CSR_TRAITS_SVH",
        "`define I3C_CSR_TRAITS_SVH",
        ""
    ]

    for cfg in CONFIGS:
        out_lines.append(f"// ---------------------------------------------------------")
        out_lines.append(f"// Class: {cfg['class_name']}")
        out_lines.append(f"// ---------------------------------------------------------")
        out_lines.append(f"class {cfg['class_name']};")

        pkg_path = Path(source_dir) / cfg["pkg_file"]
        
        if not pkg_path.exists():
            print(f"Warning: {cfg['pkg_file']} not found in {source_dir}. Generating empty class.")
            out_lines.append("endclass\n")
            continue
            
        # Read the generated package to see which structs actually exist
        content = pkg_path.read_text(encoding="utf-8")
        
        for suffix, generic_name in TRAITS_MAP.items():
            expected_struct_name = f"{cfg['prefix']}{suffix}"
            
            # Check if this exact struct was generated in this specific package
            # Example: looking for "controller_and_target_I3CCSR__DAT__out_t"
            if expected_struct_name in content:
                out_lines.append(f"  typedef {cfg['pkg_name']}::{expected_struct_name:<60} {generic_name};")
            else:
                combo_struct = f"controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR{suffix}"
                out_lines.append(f"  typedef {combo_struct:<60} {generic_name}; // DUMMY (Borrowed from combo for parser)")
        
        out_lines.append("endclass\n")

    out_lines.append("`endif // I3C_CSR_TRAITS_SVH\n")

    Path(output_file).write_text("\n".join(out_lines))
    print(f"Successfully generated {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_csr_types.py <path_to_csr_script_output_dir>")
        sys.exit(1)
        
    source_dir = sys.argv[1]
    
    # Save the output file in the same directory as the packages
    output_file = Path(source_dir) / "csr_types.svh"
    
    generate_csr_types(source_dir, output_file)

if __name__ == "__main__":
    main()
