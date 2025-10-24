import json


def calculate_average_from_file(filepath):
    total_end_to_end = 0
    count = 0
    try:
        with open(filepath, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    print(data)
                    if (
                        "end_to_end" in data
                        and isinstance(data["end_to_end"], (int, float))
                        and data["iter_idx"] != 1
                    ):
                        total_end_to_end += data["end_to_end"]
                        count += 1
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from line: {line.strip()}")
                except KeyError:
                    print(
                        f"Warning: 'end_to_end' key not found in line: {line.strip()}"
                    )
                except TypeError:
                    print(
                        "Warning: 'end_to_end' value is not a number in line:"
                        f"{line.strip()}"
                    )
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return 0

    if count == 0:
        return 0
    return total_end_to_end / count


average_from_file = calculate_average_from_file(
    "result/layer_split/Qwen/Qwen3-32B-50ms/layer_split_head.jsonl"
)
print(f"Average end_to_end from file: {average_from_file}")
