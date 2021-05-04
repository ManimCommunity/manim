import itertools as it


print("\nRendering Summary\n-----------------\n")
with open("rendering_times.csv") as file:
    data = [line.split(",") for line in file.read().splitlines()]
    # print(data)
    max_file_length = max([len(row[0]) for row in data])
    for key, group in it.groupby(data, key=lambda row: row[0]):
        group = list(group)
        time_sum = sum([float(row[2]) for row in group])
        print(f"{key.ljust(max_file_length)} {f'{time_sum:.3f}'.rjust(7)}")
        for row in group:
            print(f"{' '*max_file_length} {row[2].rjust(7)} {row[1]}")

    # print(data)

## Python program to print the data
# d = {
#     1: ["Python", 33.2, "UP"],
#     2: ["Java", 23.54, "DOWN"],
#     3: ["Ruby", 17.22, "UP"],
#     10: ["Lua", 10.55, "DOWN"],
#     5: ["Groovy", 9.22, "DOWN"],
#     6: ["C", 1.55, "UP"],
# }
# print("{:<8} {:<15} {:<10} {:<10}".format("Pos", "Lang", "Percent", "Change"))
# for k, v in d.items():
#     lang, perc, change = v
#     print("{:<8} {:<15} {:<10} {:<10}".format(k, lang, perc, change))


# examples,ManimCELogo,0.083
# examples,BraceAnnotation,0.207
# examples,VectorArrow,0.089
# examples,GradientImageFromArray,0.047


# examples                     1.123:
#                              0.083 ManimCELogo
#                              0.207 BraceAnnotation
#                              0.089 VectorArrow

# manim.utils.tex_templates   15.123:
