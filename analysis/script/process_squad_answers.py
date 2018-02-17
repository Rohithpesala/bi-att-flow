import os, json


def convert_answers(answer):
    final = {}
    for key, value in answer.items():
        if (type(key) == 'str' or type(value) == 'str'):
            print(type(key), type(value))
        final[key] = value
    return final


def main():
    # print(os.getcwd())

    answer_file = open(
        "../data/squad_data/answer-test-009000.json")
    train_file = open("../data/squad_data/train-v1.1.json")
    dev_file = open(
        "../data/squad_data/dev-v1.1.json")

    answer = json.load(answer_file)
    train = json.load(train_file)
    dev = json.load(dev_file)

    # print_dataset(answer)
    converted_answers = convert_answers(answer)
    # print(list(converted_answers.keys())[0])
    # print_dataset_full(train)
    # print_dataset_full(dev)
    # print(answer)
    # compared_json = iterate_over_answers(converted_answers, train)
    compared_json = iterate_over_answers(converted_answers, dev)
    print(len(compared_json))

    output_json = open("output.json", 'w')
    json.dump(compared_json, output_json)
    output_json.close()

    output_csv = open("output.csv", "w",encoding="utf-8")
    for comparision_dict in compared_json:
        line = ""

        for key, value in comparision_dict.items():
            entry = str(value)
            if type(value) is list:
                    entry = " : ".join(value)
            line += entry+","

        line = line[:-1]
        # print(line)
        output_csv.write(line + "\n")
    output_csv.close()

def get_answers(id, answers):
    for key, value in answers.items():
        if key == id:
            return value


def iterate_over_answers(answers, train):
    final_dict = []
    predicted_ans_dict = answers
    answer_keys_list = list(answers.keys())
    data = train['data']
    for title_dict in data:
        title = title_dict['title']
        paragraphs = title_dict['paragraphs']
        for context_dict in paragraphs:
            context = context_dict['context']
            qas_dict = context_dict['qas']
            for qas in qas_dict:
                question = qas['question']
                id = qas['id']
                if (id in answer_keys_list):
                    # print(question)
                    current_dict = {}
                    current_dict['question'] = question.replace(',', ' ')
                    predicted_answer = get_answers(id, predicted_ans_dict)
                    if (',' in predicted_answer):
                        # print("removing commas")
                        corrected_answer = predicted_answer.replace(',',' ')
                        # print(predicted_answer,"--", corrected_answer)
                        current_dict['predicted_answer'] = corrected_answer
                    else:
                        current_dict['predicted_answer'] = predicted_answer

                    answers = []
                    current_dict['exact_match'] = 0
                    current_dict['substring_match'] = 0
                    for answer_dict in qas['answers']:
                        potential_answer = answer_dict['text']
                        answers.append(potential_answer)
                        if predicted_answer in potential_answer:
                            current_dict['substring_match'] = 1
                        if predicted_answer == potential_answer:
                            current_dict['exact_match'] = 1
                    current_dict['correct_answers'] = answers
                    # print(current_dict)
                    final_dict.append(current_dict)
    print("DONE")
    return final_dict


def print_dataset(answer):
    count = 10
    for key, value in answer.items():
        print(key, value)
        count -= 1
        if (count == 0):
            break


## For each data set out of:
## train-v1.1 and dev-v1.1, they are in the format:
## {
# "data":
# [{
#   "title":
#   "paragraphs": [
#       {
#           "context":
#           "qas": [{
#                 "answer":
#                 "question":
#                 "id":
#                  },
#                  ...,
#                  {}
#                 ]
#       },
#       {}
#   ]
# }.
# {},
# "version":
## }
##
def print_dataset_full(data):
    data = data['data']
    # print(data[0]['paragraphs'])
    single_paragraph = data[0]['paragraphs'][0]
    # print(single_paragraph['context'])
    print(single_paragraph['qas'])


if __name__ == '__main__':
    main()
