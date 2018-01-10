from sys import argv

if len(argv) != 4:
  print("Usage: python create_indel_locations <chromosome_number> <include_filtered [0 or 1]> <differentiate_insertions_and_deletions [0 or 1]>")
  exit()

chromosome_id = argv[1]
include_f = int(argv[2])
diff_t = int(argv[3])

filename = "/datadrive/project_data/gnomad_indels.tsv"

if diff_t:
  ins_locations = []
  del_locations = []
else:
  locations = []

with open(filename) as f:
  for i, line in enumerate(f):
    if i % 1000000 == 0: print("Line {}...".format(i))
    if i == 0: continue
    l = line.split()
    chromosome = l[0]
    location = int(l[1]) # The position is 1-indexed! We should do - 1 here?
    if chromosome == chromosome_id and (include_f or l[4] != 'true'):
      if diff_t:
        if len(l[3]) > len(l[2]): # 'Alt' longer than 'ref': insertion
          ins_locations.append(location)
        else:
          del_locations.append(location)
      else:
        locations.append(location)

outfile = "/datadrive/project_data/indelLocations" + chromosome_id
if not include_f: outfile += "_filtered"
if diff_t:
  with open(outfile + "_ins.txt", 'w') as f:
    for l in ins_locations:
      f.write(str(l) + '\n')
  with open(outfile + "_del.txt", 'w') as f:
    for l in del_locations:
      f.write(str(l) + '\n')
else:
  with open(outfile + ".txt", 'w') as f:
    for l in locations:
      f.write(str(l) + '\n')
