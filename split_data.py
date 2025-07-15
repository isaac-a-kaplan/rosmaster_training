import os

def split_jsonl_file(input_path, max_size_mb=95):
    max_bytes = max_size_mb * 1024 * 1024
    part_num = 1
    current_bytes = 0
    output = open(f"prepared_data/{input_path}_part{part_num}.jsonl", "w", encoding="utf-8")

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line_size = len(line.encode("utf-8"))
            if current_bytes + line_size > max_bytes:
                output.close()
                part_num += 1
                current_bytes = 0
                output = open(f"prepared_data/{input_path}_part{part_num}.jsonl", "w", encoding="utf-8")
            output.write(line)
            current_bytes += line_size

    output.close()
    print(f"Split into {part_num} files.")

# Usage:
split_jsonl_file("prepared_data.jsonl")
