import sys
from pathlib import Path
import jinja2

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


def generate_csr_types(source_dir: str, output_file: Path, template_dir: Path):
    templateLoader = jinja2.FileSystemLoader(searchpath=template_dir)
    templateEnv = jinja2.Environment(loader=templateLoader)
    
    TEMPLATE_FILE = "csr_types_template.j2"
    try:
        template = templateEnv.get_template(TEMPLATE_FILE)
    except jinja2.exceptions.TemplateNotFound:
        print(f"Error: Template file '{TEMPLATE_FILE}' not found in '{template_dir}'")
        sys.exit(1)

    template_data = []

    for cfg in CONFIGS:
        pkg_path = Path(source_dir) / cfg["pkg_file"]
        
        cfg_data = {
            "class_name": cfg["class_name"],
            "is_empty": False,
            "typedefs": []
        }

        if not pkg_path.exists():
            print(f"Warning: {cfg['pkg_file']} not found in {source_dir}. Generating empty class.")
            cfg_data["is_empty"] = True
            template_data.append(cfg_data)
            continue
            
        # Read the generated package to see which structs actually exist
        content = pkg_path.read_text(encoding="utf-8")
        
        for suffix, generic_name in TRAITS_MAP.items():
            expected_struct_name = f"{cfg['prefix']}{suffix}"
            
            # Check if this exact struct was generated in this specific package
            if expected_struct_name in content:
                source_type = f"{cfg['pkg_name']}::{expected_struct_name}"
                is_dummy = False
            else:
                source_type = f"controller_and_target_I3CCSR_pkg::controller_and_target_I3CCSR{suffix}"
                is_dummy = True
                
            cfg_data["typedefs"].append({
                "source_type": source_type,
                "generic_name": generic_name,
                "is_dummy": is_dummy
            })
            
        template_data.append(cfg_data)

    # Render the template with the prepared data
    rendered_output = template.render(configs=template_data)
    output_file.write_text(rendered_output)
    print(f"Successfully generated {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_csr_types.py <path_to_csr_script_output_dir>")
        sys.exit(1)
        
    source_dir = sys.argv[1]
    
    # Save the output file in the same directory as the packages
    output_file = Path(source_dir) / "csr_types.svh"

    script_dir = Path(__file__).parent.resolve()
    
    generate_csr_types(source_dir, output_file, template_dir=script_dir)

if __name__ == "__main__":
    main()
