import itertools as it

print("\nRendering Summary\n-----------------\n")
with open("rendering_times.csv") as file:
    data = [line.split(",") for line in file.read().splitlines()]
    # print(data)
    max_file_length = max([len(row[0]) for row in data])
    for key, group in it.groupby(data, key=lambda row: row[0]):
        key = key.ljust(max_file_length + 1, ".")
        group = list(group)
        if len(group) == 1:
            row = group[0]
            print(f"{key}{row[2].rjust(7, '.')}s {row[1]}")
            continue
        time_sum = sum([float(row[2]) for row in group])
        print(f"{key}{f'{time_sum:.3f}'.rjust(7, '.')}s  => {len(group)} EXAMPLES")
        for row in group:
            print(f"{' '*(max_file_length)} {row[2].rjust(7)}s {row[1]}")
