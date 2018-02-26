import pandas as pd
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", default='./split_data/test.csv')
    parser.add_argument("--target_file", default='test-v1.1.json')
    args = parser.parse_args()
    df = pd.read_csv(args.source_file)
    list_id = df['story_id'].unique()
    data = []
    # count = 0
    for id in list_id:
        # if count > 0:
        #     break
        # count += 1

        paragraph = []

        df_short = df.ix[df['story_id'] == id]
        story_id = id.split("/")[-1].split('.')[0]

        qas = []
        story = df_short['story_text'].values[0].split(' ')
        i = 0
        for index, row in df_short.iterrows():
            answers = row['answer_token_ranges'].replace(',', "|")
            answers = answers.split('|')

            ans = []
            flag = 0
            for entity in answers:
                if entity != 'None':
                    flag = 1
                    rangeis = entity.split(":")
                    answer = story[int(rangeis[0]):int(rangeis[1])]

                    startindex = 0
                    for word in story[:int(rangeis[0])]:
                        startindex += (len(word) + 1)

                    answerfinal = ' '.join(answer)
                    ans.append({"text": answerfinal, "answer_start": startindex})
                    # print("Story Id: {} , Question is: {}, Answer is: {}, Start Index: {}".format(story_id, row['question'], answer, startindex))

            if flag == 0:
                continue

            qas.append({"question": row['question'].decode('latin-1'), "id": story_id + str(i), "answers": ans})
            i += 1

        if len(qas) == 0:
            continue
        paragraph.append({"qas": qas, "context": df_short['story_text'].values[0]})
        data.append({"paragraphs": paragraph, "title": "Hello"})
    jsonfinal = {"data": data, "version": "3"}
    with open('./' + args.target_file, 'w') as fp:
        json.dump(jsonfinal, fp)

if __name__ == '__main__':
    main()
