# SPDX-License-Identifier: Apache-2.0
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This script generates SV registers and uvm classes from RDL files

import argparse
import logging
import os
from pathlib import Path

from gen_axi_csr_tracker import generate_axi_csr_tracker
from peakrdl_cheader.exporter import CHeaderExporter
from peakrdl_cocotb.exporter import CocotbExporter
from peakrdl_html import HTMLExporter
from peakrdl_markdown import MarkdownExporter
from peakrdl_regblock import RegblockExporter
from peakrdl_regblock.cpuif.passthrough import PassthroughCpuif
from peakrdl_regblock.udps import ALL_UDPS
from peakrdl_uvm import UVMExporter
from rdl_post_process import postprocess_sv
from systemrdl import RDLCompiler

def setup_logger(level=logging.INFO, filename="log.log"):
    logging.basicConfig(
        level=level, handlers=[logging.FileHandler(filename), logging.StreamHandler()]
    )

def rename_module_in_file(filepath, old_name, new_name):
    if filepath.exists():
        content = filepath.read_text()
        filepath.write_text(content.replace(old_name, new_name))

def main():
    setup_logger(level=logging.INFO, filename="reg_gen.log")

    repo_root = Path(os.environ.get("CALIPTRA_ROOT"))
    if not repo_root.exists():
        raise ValueError("Caliptra root is not defined as environment variable. Aborting.")

    def get_template_path(name):
        return repo_root / "tools" / "templates" / "rdl" / name

    parser = argparse.ArgumentParser(description="Reg gen")
    parser.add_argument(
        "--style-hier",
        action="store_true",
        help="Style: hierarchical or lexical",
        default=True,
    )
    parser.add_argument(
        "--input-file", default="./src/rdl/registers.rdl", help="input SystemRDL file"
    )
    parser.add_argument("--output-dir", default="./src/csr/script/", help="output directory")
    parser.add_argument("-P", action="append", help="SystemRDL parameters", metavar="key=value")
    parser.add_argument(
        "--ral-template", default=get_template_path("uvm"), help="Template for generating UVM RAL"
    )
    parser.add_argument(
        "--cov-template",
        default=get_template_path("cov"),
        help="Template for generating RAL coverage groups",
    )
    parser.add_argument(
        "--smp-template",
        default=get_template_path("smp"),
        help="Template for implementing sample functions for RAL coverage",
    )
    args = parser.parse_args()

    # Parse Parameters
    base_parameters = {}
    for p in args.P or []:
        try:
            p_split = p.split("=")
            text = p_split[0]
            number = int(p_split[-1])
            base_parameters[text] = number
        except Exception:
            raise ValueError(
                f"SystemRDL Parameters should be a space separated list. Expected: -P param_1=1 -P param2=2. Got: {p}"
            )
    output_dir = Path(args.output_dir)

    # Compile RDL once
    rdlc = RDLCompiler()
    for udp in ALL_UDPS:
        rdlc.register_udp(udp)

    rdlc.compile_file(args.input_file)

    # Define the 3 configurations
    configs = [
        {"prefix": "controller_I3CCSR",            "params": {"ControllerEn": 1, "TargetEn": 0}},
        {"prefix": "target_I3CCSR",                "params": {"ControllerEn": 0, "TargetEn": 1}},
        {"prefix": "controller_and_target_I3CCSR", "params": {"ControllerEn": 1, "TargetEn": 1}},
    ]

    i3c_root_dir = Path(os.environ.get("I3C_ROOT_DIR"))

    for config in configs:
        REGISTERS_PREFIX = config["prefix"]
        logging.info(f"--- Generating Configuration: {REGISTERS_PREFIX} ---")

        # Merge base CLI parameters with the config parameters
        run_params = base_parameters.copy()
        run_params.update(config["params"])

        root = rdlc.elaborate(parameters=run_params)

        # Export SystemVerilog implementation
        exporter = RegblockExporter()
        exporter.export(
            root,
            str(output_dir),
            cpuif_cls=PassthroughCpuif,
            retime_read_response=False,
            reuse_hwif_typedefs=not args.style_hier,
        )
        
        # Rename standard output files to the prefix
        sv_file = output_dir / f"{REGISTERS_PREFIX}.sv"
        pkg_file = output_dir / f"{REGISTERS_PREFIX}_pkg.sv"
        if (output_dir / "I3CCSR.sv").exists(): (output_dir / "I3CCSR.sv").rename(sv_file)
        if (output_dir / "I3CCSR_pkg.sv").exists(): (output_dir / "I3CCSR_pkg.sv").rename(pkg_file)
        
        rename_module_in_file(sv_file, "I3CCSR", REGISTERS_PREFIX)
        rename_module_in_file(pkg_file, "I3CCSR", REGISTERS_PREFIX)
        logging.info(f"Created: SystemVerilog files in {output_dir}")

        # Export UVM register model
        file_path_uvm = REGISTERS_PREFIX + "_uvm.sv"
        output_file = output_dir / file_path_uvm
        exporter = UVMExporter(user_template_dir=args.ral_template)
        exporter.export(
            root,
            str(output_file),
            reuse_class_definitions=not args.style_hier,
        )
        rename_module_in_file(output_file, "I3CCSR", REGISTERS_PREFIX)
        logging.info(f"Created: UVM file {output_file}")

        def export_uvm_collateral(template_path, collateral_suffix):
            file_path = REGISTERS_PREFIX + collateral_suffix
            output_file = output_dir / file_path
            exporter = UVMExporter(user_template_dir=template_path)
            exporter.export(
                root,
                str(output_file),
                reuse_class_definitions=not args.style_hier,
            )
            rename_module_in_file(output_file, "I3CCSR", REGISTERS_PREFIX)
            logging.info(f"Created file {output_file}")

        export_uvm_collateral(args.cov_template, "_covergroups.svh")
        export_uvm_collateral(args.smp_template, "_sample.svh")

        # Generate the C header
        exporter = CHeaderExporter()
        try:
            (i3c_root_dir / "sw").mkdir(exist_ok=True)
        except FileExistsError:
            pass
        output_file = i3c_root_dir / "sw" / (REGISTERS_PREFIX + ".h")
        exporter.export(root, path=str(output_file), reuse_typedefs=not args.style_hier)
        rename_module_in_file(output_file, "I3CCSR", REGISTERS_PREFIX.upper())
        logging.info(f"Created: c-header file {output_file}")

        # Export documentation in HTML (Protected against plugin crash)
        try:
            exporter = HTMLExporter()
            output_file = i3c_root_dir / "src" / "rdl" / "docs" / REGISTERS_PREFIX / "html"
            exporter.export(root, str(output_file))
            logging.info(f"Created: HTML files in {output_file}")
        except Exception as e:
            logging.warning(f"Skipped HTML for {REGISTERS_PREFIX} due to plugin error: {e}")

        # Export Markdown documentation (Protected against plugin crash)
        try:
            exporter = MarkdownExporter()
            output_file = i3c_root_dir / "src" / "rdl" / "docs" / REGISTERS_PREFIX / "README.md"
            exporter.export(root, str(output_file), rename=REGISTERS_PREFIX)
            logging.info(f"Created: Markdown file {output_file}")
        except Exception as e:
            logging.warning(f"Skipped Markdown for {REGISTERS_PREFIX} due to plugin error: {e}")

        # Fix SystemVerilog files
        postprocess_sv(output_dir / (REGISTERS_PREFIX + ".sv"))
        postprocess_sv(output_dir / (REGISTERS_PREFIX + "_pkg.sv"))

        # Export Cocotb dictionary
        exporter = CocotbExporter()
        output_file = i3c_root_dir / "verification" / "cocotb" / "common" / f"reg_map_{REGISTERS_PREFIX}.py"
        exporter.export(root, path=str(output_file))
        logging.info(f"Created: Python dictionary file {output_file}")

        # Generate AXI CSR tracker bind module
        import importlib.util
        spec = importlib.util.spec_from_file_location("reg_map", str(output_file))
        reg_map_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reg_map_mod)
        tracker_output = (
            i3c_root_dir / "verification" / "cocotb" / "top" / "lib_i3c_top" / f"axi_csr_tracker_{REGISTERS_PREFIX}.sv"
        )
        generate_axi_csr_tracker(reg_map_mod.reg_map, tracker_output)

if __name__ == "__main__":
    main()
