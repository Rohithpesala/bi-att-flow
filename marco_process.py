import json
import sys
import pandas as pd
import pickle
from rouge import Rouge 
import os
rouge = Rouge()

for i in range(len(sys.argv)-1):
	fname = sys.argv[i+1]
	nf = []
	with open(os.getcwd()+fname) as f:
		for l in f:
			nl = l.replace("\n",",")
			nf.append(nl)
	with open("clean.json","w") as vlc_file:
		vlc_file.write("{\"data\":["+("".join(nf))[:-1]+"]}")
	
	jf = pd.read_json('clean.json')
	# jf['version'] = 1.0
	data = []
	for i,di in enumerate(jf['data']):
		# print(di)
		query_passage = ""
		query_id = di['query_id']
		query = di['query']
		title = di['query_type']
		answers = []
		c = 0
		# # print "=================================================="
		for j,pj in enumerate(di['passages']):
			query_passage += ' '.join(pj['passage_text'].split()) + ' '
			if pj['is_selected'] == 1:
				c+=1
				# print pj['passage_text']
				for k,ans in enumerate(di['answers']):
					temp_passage = pj['passage_text'].split()
					max_score = 0.0
					max_start  = 0
					max_end = 1
					# print temp_passage
					for start in range(len(temp_passage)):
						for end in range(start+1,len(temp_passage)+1):
							# print(ans)
							# print((str(u' '.join(temp_passage[start:end]).encode('utf-8').strip())))
							score = rouge.get_scores(ans, u' '.join(temp_passage[start:end]))[0]['rouge-l']['f']
							# print(start,end,i)
							# print(score)
							if score > max_score:
								max_score = score
								max_start = start
								max_end = end
						# 	break
						# 	pass
						# break
					answer_text = u' '.join(temp_passage[max_start:max_end])
					answer_start = len(query_passage)- len(' '.join(pj['passage_text'].split())) + len(u' '.join(temp_passage[0:max_start]))
					if max_start == 0:
						answer_start -=1
						if max_end == 1:
							continue
					answer_temp = {"text": answer_text, "answer_start": answer_start}
					answers.append(answer_temp)
					# print(answers)
					# break
					# scores = rouge.get_scores(ans, hypothesis)
			# break
		if len(answers) == 0:
			continue
		# print(di['answers'])
		# break
		# print(answers)
		qas = [{"question": query, "id": query_id, "answers": answers}]
		para_temp = [{"qas": qas, "context": query_passage}]
		data_temp = {"paragraphs":para_temp, "title": title}
		data.append(data_temp)
		print(i)
		# if i>100:
		# 	break
	nf = {"data": data, "version": 1.0}
	json.dump(nf, open(os.getcwd()+fname, 'w'))

