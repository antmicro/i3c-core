#!/usr/bin/env python3
import os
import re
import sys
from pathlib import Path
import ast

def get_python_tests(cocotb_dir):
    """Finds all functions decorated with @cocotb.test OR passed to TestFactory."""
    test_funcs = set()
    
    for root, _, files in os.walk(cocotb_dir):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                path = Path(root) / file
                
                if path.is_symlink():
                    continue
                
                with open(path, 'r', encoding='utf-8') as f:
                    try:
                        tree = ast.parse(f.read(), filename=str(path))
                    except SyntaxError:
                        continue 
                    
                    for node in ast.walk(tree):
                        # 1. Look for standard @cocotb.test decorators
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if node.name.startswith('test_'):
                                is_test_decorated = False
                                for dec in node.decorator_list:
                                    dec_node = dec.func if isinstance(dec, ast.Call) else dec
                                    if isinstance(dec_node, ast.Name) and dec_node.id == 'test':
                                        is_test_decorated = True
                                    elif isinstance(dec_node, ast.Attribute) and dec_node.attr == 'test':
                                        is_test_decorated = True
                                        
                                if is_test_decorated:
                                    test_funcs.add(node.name[5:])
                                    
                        # 2. NEW: Look for TestFactory(...) calls
                        elif isinstance(node, ast.Call):
                            if isinstance(node.func, ast.Name) and node.func.id == 'TestFactory':
                                # Check for keyword argument: test_function=test_name
                                for kw in node.keywords:
                                    if kw.arg == 'test_function' and isinstance(kw.value, ast.Name):
                                        func_name = kw.value.id
                                        if func_name.startswith('test_'):
                                            test_funcs.add(func_name[5:])
                                            
                                # Check for positional argument: TestFactory(test_name)
                                if node.args and isinstance(node.args[0], ast.Name):
                                    func_name = node.args[0].id
                                    if func_name.startswith('test_'):
                                        test_funcs.add(func_name[5:])
                                        
    return test_funcs

def get_hjson_tests(testplan_dir):
    """Finds all test names listed in the tests: ["..."] arrays of HJSON files."""
    hjson_tests = set()
    # ADDED re.DOTALL: Allows (.*?) to match across multiple lines
    pattern = re.compile(r'tests:\s*\[(.*?)\]', re.DOTALL)
    
    for root, _, files in os.walk(testplan_dir):
        for file in files:
            if file.endswith('.hjson'):
                path = Path(root) / file
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    matches = pattern.finditer(content)
                    for match in matches:
                        tests_str = match.group(1)
                        # Extract everything inside quotes within the array brackets
                        test_names = re.findall(r'"([^"]+)"', tests_str)
                        hjson_tests.update(test_names)
    return hjson_tests

def main():
    i3c_root = os.getenv("I3C_ROOT_DIR")
    
    if i3c_root:
        root_dir = Path(i3c_root).resolve()
    else:
        # Safe fallback for local developers who forgot to export the variable
        root_dir = Path.cwd()
        print(f"⚠️ WARNING: I3C_ROOT_DIR environment variable not set. Defaulting to {root_dir}\n")

    cocotb_dir = root_dir / "verification" / "cocotb"
    testplan_dir = root_dir / "verification" / "testplan"

    print("🔍 Linting Testplans vs Implementation (ignoring 'test_' prefix)...")
    
    py_tests = get_python_tests(cocotb_dir)
    hjson_tests = get_hjson_tests(testplan_dir)

    undocumented = py_tests - hjson_tests
    ghost_tests = hjson_tests - py_tests

    errors = 0

    if undocumented:
        print("\n❌ ERROR: Found implemented tests with NO testplan documentation:")
        print("   (Note: the 'test_' prefix has been omitted below)")
        for t in sorted(undocumented):
            print(f"  - {t}")
        errors += 1

    if ghost_tests:
        print("\n❌ ERROR: Found documented tests in HJSON that do NOT exist in Python:")
        for t in sorted(ghost_tests):
            print(f"  - {t}")
        errors += 1

    if errors == 0:
        print("\n✅ SUCCESS: All Python tests and HJSON testplans are perfectly synced!")
        sys.exit(0)
    else:
        print("\n🚨 Please update your .hjson files to match your Python code.")
        sys.exit(1)

if __name__ == "__main__":
    main()
