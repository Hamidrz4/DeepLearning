import json

def jskeys(file_path):
    def print_tree(data, level=0, keys_printed=None):
        if keys_printed is None:
            keys_printed = set()

        if isinstance(data, dict):
            for key, value in data.items():
                if key not in keys_printed:
                    print("  " * level + key)
                    keys_printed.add(key)
                print_tree(value, level + 1, keys_printed)
        elif isinstance(data, list):
            for item in data:
                print_tree(item, level, keys_printed)

    try:
        with open(file_path, 'r') as json_file:
            data_dict = json.load(json_file)

        print_tree(data_dict)

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Invalid JSON format in file: {file_path}")



