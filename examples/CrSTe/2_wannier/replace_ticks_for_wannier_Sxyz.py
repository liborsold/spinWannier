def get_lines_starting_with_tick(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        tick_lines = [line for line in lines if line.startswith("tick")]
        return tick_lines

ticks_from_file = 'wannier90-bands.py'
ticks_to_file = 'wannier90-bands_Sxyz.py'

string_to_replace = "    #ticks"

# get all lines starting with "tick"
ticks = get_lines_starting_with_tick(ticks_from_file)

# format the string
tick_string = "    " + "    ".join(ticks)

# replace the string
with open(ticks_to_file, 'r') as fr:
    text_Sxyz = fr.read()
    text_Sxyz = text_Sxyz.replace(string_to_replace, tick_string)

# write the result
with open(ticks_to_file, 'w') as fw:
    fw.write(text_Sxyz)
