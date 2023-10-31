import diaparser.utils.data as data

with open(data.__file__, "r") as file:
    source = file.read()
    if "__getitems__" in source:
        print("Diaparser module is already hotfixed.")
        exit()
    source = source.splitlines()

source.insert(62, "        if name == '__getitems__':")
source.insert(63, "            return None")

print('\n'.join(source[61:70]))

with open(data.__file__, "w") as file:
    file.write("\n".join(source))