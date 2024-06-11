# Open the file in read mode and read lines
with open('../data/expe-data-grenoble-new.rawdata', 'r') as file:
    lines = file.readlines()

# Find the first line that is a line jump
line_jump_index = next((i for i, line in enumerate(lines) if line.strip(';') == '\n'), None)

if line_jump_index is not None:
    print(f"The first line jump is at line {line_jump_index + 1}")
else:
    print("No line jump found in the file.")

print(line_jump_index)
# Process lines starting from the line after the first line jump
for i, line in enumerate(lines[line_jump_index+1:], start=line_jump_index+1):
    # If line is a line jump, stop processing
    if line.strip(';') == '\n':
        break

    # Split the line into elements
    elements = line.split(';')

    # Subtract 3694.0 from the first element and replace it
    elements[0] = str(float(elements[0]) - 20686.0)

    # Join the elements back into a line and replace the line
    lines[i] = ';'.join(elements)

# Open the file in write mode and write lines
with open('../data/expe-data-grenoble-new.rawdata', 'w') as file:
    file.writelines(lines)