import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, required=True)
parser.add_argument("--prefix", type=str, required=True)
args = parser.parse_args()

line_start = len(args.prefix) + 1

for file in os.listdir(args.dir):
    output_pth = os.path.join(args.dir, f"{file}.raw")
    in_pth = os.path.join(args.dir, file)
    print(f"{file} -> {file}.raw")

    with open(in_pth) as inf, open(output_pth, "w") as ouf:
        buffer = []
        for line in inf:
            line = line.strip()
            if not line:
                output = " ".join(buffer)
                ouf.write(f"{output}\n")
                buffer = []
            else:
                line = line[line_start:]
                word = line.split("\t")[0]
                buffer.append(word)
        if buffer:
            output = " ".join(buffer)
            ouf.write(f"{output}\n")

