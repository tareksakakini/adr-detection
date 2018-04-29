import re

def seq2ent(seq):
        ents = []
        ent_start_flag = False
        ent_type = None
        ent_start = 0
        ent_end = 0
        for i,tag in enumerate(seq):
                if tag[0] == "B":
                        ent_start = i
                        ent_type = tag[-1]
                        ent_start_flag = True
                elif tag == "O" and ent_start_flag:
                        ents.append((ent_type,ent_start,i))
                        ent_start_flag = False
        return set(ents)

infile_path_ner = "/home/sakakini/adr-detection-parent/large-files/datasets/ADE/ADE_NER_All.txt"
infile_path_pos = "ADE-tagged-POS.txt"

lines_ner = open(infile_path_ner).readlines()
lines_pos = open(infile_path_pos).readlines()

pos_seqs = []

for i in range(len(lines_pos)):
	[in_sent_pos, pos_tags] = lines_pos[i].split("\t")
	[in_sent_ner, ner_tags] = lines_ner[i].split("\t")
	in_sent_pos = in_sent_pos.split()
	in_sent_ner = in_sent_ner.split()
	pos_tags = pos_tags.split()
	ner_tags = ner_tags.split()
	if in_sent_pos != in_sent_ner:
		print "Warning: mismatch"
		continue
	ents = seq2ent(ner_tags) 
	for ent in ents:
		(t, start, end) = ent
		pos_seqs.append((t, in_sent_pos[start:end], pos_tags[start:end]))

for seq in pos_seqs:
	pos_tags = " ".join(seq[2])
	key = re.compile("[(NN)(JJ)]")
	if key.search(pos_tags)!=None:
		print "Yes"
		print pos_tags
	else:
		print "No"
		print pos_tags
