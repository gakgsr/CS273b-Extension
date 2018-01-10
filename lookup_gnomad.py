from sys import argv

search_loc = argv[1]

filename = "/datadrive/project_data/gnomad_indels.tsv"

with open(filename) as f:
  for i, line in enumerate(f):
    if i % 1000000 == 0: print("Line {}...".format(i))
    if i == 0: continue
    line = line.strip()
    l = line.split()
    location = l[1]
    if location == search_loc:
      print(line)
