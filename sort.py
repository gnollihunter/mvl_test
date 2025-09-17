# Read the content of the file
with open('list.txt', 'r') as file:
    lines = file.readlines()

# Sort the lines
sorted_lines = sorted(lines)

# Write the sorted lines to a new file
with open('sorted.txt', 'w') as new_file:
    new_file.writelines(sorted_lines)
