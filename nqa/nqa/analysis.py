import os
import json


def analyze_newsqa():
    # file = open("train-v1.1.json", 'r')
    # file = open("dev-v1.1.json", 'r')
    # file = open("test-v1.1.json", 'r')
    file = open("squad/train-v1.1.json", 'r')
    f = json.load(file)
    # print(len(f['data']))
    data = f['data']
    # print(len(data[0]['paragraphs']))
    print(len(data))
    context_len = 0
    data_count = 0
    max = -1
    min = 100000
    for item in data:
        paragraphs = item['paragraphs']
        for p in paragraphs:
            context = p['context']
            curr_context_len = len(context.split(" "))

            if max < curr_context_len:
                max = curr_context_len
            if min > curr_context_len:
                min = curr_context_len
            context_len += curr_context_len
            data_count+=1
    print(float(context_len /data_count ), max, min)


def analyze_squad():
    file = open("squad/train-v1.1.json", 'r')
    f = json.load(file)
    data = f['data']
    print(len(data[0]['paragraphs'][0].keys()))
    print(data[0]['paragraphs'][0].keys())
    # print(data[0])
    context_len = 0
    max = -1
    min = 100000

    # for item in data:
    #     context = item['paragraphs'][0]['context']
    #     curr_context_len = len(context.split(" "))
    #
    #     if max < curr_context_len:
    #         max = curr_context_len
    #     if min > curr_context_len:
    #         min = curr_context_len
    #     context_len += curr_context_len
    # print(float(context_len / len(data)), max, min)


if __name__ == '__main__':
    analyze_newsqa()
    # analyze_squad()

