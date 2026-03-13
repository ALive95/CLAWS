# MAP script: maps the structure of the project
# Useful for restructuring

import os

for root, dirs, files in os.walk("."):
    level = root.replace(".", "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    for file in files:
        print(f"{indent}  {file}")