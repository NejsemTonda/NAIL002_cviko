#!/usr/bin/env python3
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Path to the Jupyter notebook file")
    args = parser.parse_args()

    name = args.filename

    with open(name, 'r', encoding='utf-8') as file:
        notebook = json.load(file)

    py_filename = name.rsplit('.', 1)[0] + '.py'

    with open(py_filename, 'w', encoding='utf-8') as py_file:
        for cell in notebook.get('cells', []):
            cell_type = cell.get('cell_type', '')
            source = ''.join(cell.get('source', []))

            if cell_type == 'code':
                py_file.write('\n' + source + '\n')
            elif cell_type in ['markdown', 'text']:
                text = source.replace('\\', '\\\\')
                py_file.write(f'\n"""\n{text}\n"""\n')

    print(f"✅ Converted '{name}' → '{py_filename}'")

if __name__ == "__main__":
    main()
