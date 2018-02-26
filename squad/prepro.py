import argparse
import json
import os
# data: q, cq, (dq), (pq), y, *x, *cx
# shared: x, cx, (dx), (px), word_counter, char_counter, word2vec
# no metadata
from collections import Counter

from tqdm import tqdm

from squad.utils import get_word_span, get_word_idx, process_tokens


def main():
    args = get_args()
    prepro(args)


def get_args():
    parser = argparse.ArgumentParser()
    # home = os.path.expanduser("~")
    # source_dir = os.path.join(home, "data", "squad")
    # target_dir = "data/squad"

    home = os.path.expanduser(".")
    source_dir = os.path.join(home, "nqa", "nqa")
    target_dir = "data/newsqa"

    glove_dir = os.path.join(home, "data", "glove")
    parser.add_argument('-s', "--source_dir", default=source_dir)
    parser.add_argument('-t', "--target_dir", default=target_dir)
    parser.add_argument('-d', "--debug", action='store_true')
    parser.add_argument("--train_ratio", default=0.9, type=int)
    parser.add_argument("--glove_corpus", default="6B")
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--glove_vec_size", default=100, type=int)
    parser.add_argument("--mode", default="full", type=str)
    parser.add_argument("--single_path", default="", type=str)
    parser.add_argument("--tokenizer", default="PTB", type=str)
    parser.add_argument("--url", default="vision-server2.corp.ai2", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--split", action='store_true')
    # TODO : put more args here
    return parser.parse_args()


def create_all(args):
    out_path = os.path.join(args.source_dir, "all-v1.1.json")
    if os.path.exists(out_path):
        return
    train_path = os.path.join(args.source_dir, "train-v1.1.json")
    train_data = json.load(open(train_path, 'r'))
    dev_path = os.path.join(args.source_dir, "dev-v1.1.json")
    dev_data = json.load(open(dev_path, 'r'))
    train_data['data'].extend(dev_data['data'])
    print("dumping all data ...")
    json.dump(train_data, open(out_path, 'w'))


def prepro(args):
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    if args.mode == 'full':
        prepro_each(args, 'train', out_name='train')
        prepro_each(args, 'dev', out_name='dev')
        prepro_each(args, 'dev', out_name='test')
    elif args.mode == 'all':
        create_all(args)
        prepro_each(args, 'dev', 0.0, 0.0, out_name='dev')
        prepro_each(args, 'dev', 0.0, 0.0, out_name='test')
        prepro_each(args, 'all', out_name='train')
    elif args.mode == 'single':
        assert len(args.single_path) > 0
        prepro_each(args, "NULL", out_name="single", in_path=args.single_path)
    else:
        prepro_each(args, 'train', 0.0, args.train_ratio, out_name='train')
        prepro_each(args, 'train', args.train_ratio, 1.0, out_name='dev')
        prepro_each(args, 'dev', out_name='test')


def save(args, data, shared, data_type):
    data_path = os.path.join(args.target_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(args.target_dir, "shared_{}.json".format(data_type))
    print("Saving data in {}".format(data_path))
    print("Saving shared data in {}".format(shared_path))
    json.dump(data, open(data_path, 'w'))
    json.dump(shared, open(shared_path, 'w'))


def get_word2vec(args, word_counter):
    glove_path = os.path.join(args.glove_dir, "glove.{}.{}d.txt".format(args.glove_corpus, args.glove_vec_size))
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes[args.glove_corpus]
    word2vec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh, total=total):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
            elif word.capitalize() in word_counter:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in word_counter:
                word2vec_dict[word.upper()] = vector

    print(
    "{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
    return word2vec_dict


def prepro_each(args, data_type, start_ratio=0.0, stop_ratio=1.0, out_name="default", in_path=None):
    if args.tokenizer == "PTB":
        print("PTB")
        import nltk
        sent_tokenize = nltk.sent_tokenize

        def word_tokenize(tokens):
            return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
    elif args.tokenizer == 'Stanford':
        print("Stanford")
        from my.corenlp_interface import CoreNLPInterface
        interface = CoreNLPInterface(args.url, args.port)
        sent_tokenize = interface.split_doc
        word_tokenize = interface.split_sent
    else:
        raise Exception()

    if not args.split:
        sent_tokenize = lambda para: [para]

    source_path = in_path or os.path.join(args.source_dir, "{}-v1.1.json".format(data_type))
    print("Source path being loaded into source_data: ", source_path)
    source_data = json.load(open(source_path, 'r'))

    q, cq, y, rx, rcx, ids, idxs = [], [], [], [], [], [], []
    cy = []
    x, cx = [], []
    answerss = []
    p = []
    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
    start_ai = int(round(len(source_data['data']) * start_ratio))
    stop_ai = int(round(len(source_data['data']) * stop_ratio))
    article_index = -12
    assert_fail_count = 0
    len_fail_count = 0
    for ai, article in enumerate(tqdm(source_data['data'][start_ai:stop_ai])):
        if (ai == article_index):
            print("AI,ARTICLE", ai, article)
        xp, cxp = [], []
        pp = []
        x.append(xp)
        cx.append(cxp)
        p.append(pp)
        for pi, para in enumerate(article['paragraphs']):
            # wordss
            if (ai == article_index):
                print(pi, para)
            context = para['context']
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')
            xi = list(map(word_tokenize, sent_tokenize(context)))
            xi = [process_tokens(tokens) for tokens in xi]  # process tokens
            # given xi, add chars
            if (ai == article_index):
                print("\n--------XI: ", xi)
            cxi = [[list(xijk) for xijk in xij] for xij in xi]
            if (ai == article_index):
                print("\n--------CXI: ", cxi)
            xp.append(xi)
            if (ai == article_index):
                print("\n--------XP: ", xp)
            cxp.append(cxi)
            if (ai == article_index):
                print("\n--------CXP: ", cxp)
            pp.append(context)
            if (ai == article_index):
                print("\n--------PP: ", pp)

            for xij in xi:
                for xijk in xij:
                    word_counter[xijk] += len(para['qas'])
                    lower_word_counter[xijk.lower()] += len(para['qas'])
                    for xijkl in xijk:
                        char_counter[xijkl] += len(para['qas'])

            rxi = [ai, pi]
            assert len(x) - 1 == ai
            assert len(x[ai]) - 1 == pi
            if (ai == article_index):
                print("\n--------QAS: ", para['qas'])
            for qa in para['qas']:
                # get words
                qi = word_tokenize(qa['question'])
                # if (ai==article_index):
                #     print("\n--------QI: ", qi)
                cqi = [list(qij) for qij in qi]
                # if (ai==article_index):
                #     print("\n--------CQI: ", cqi)
                yi = []
                cyi = []
                answers = []
                for answer in qa['answers']:
                    answer_text = answer['text']
                    answer_text = answer_text.replace("''", '"').replace("``", '"')
                    answers.append(answer_text)
                    answer_start = answer['answer_start']
                    answer_stop = answer_start + len(answer_text)
                    if (ai == article_index):
                        print("\n--------Answer Text: ", answer_text)
                        print("\n--------Len Answer Text: ", len(answer_text))
                        print("\n--------Answers: ", answers)
                        print("\n--------Answers start, stop: ", answer_start, answer_stop)

                    # TODO : put some function that gives word_start, word_stop here
                    yi0, yi1 = get_word_span(context, xi, answer_start, answer_stop)
                    if (yi0 == -1 and yi1 == -1):
                        print("Continuing on")
                        continue
                    # yi0 = answer['answer_word_start'] or [0, 0]
                    # yi1 = answer['answer_word_stop'] or [0, 1]
                    if (ai == article_index):
                        print("\n--------YI0, YI1: ", yi0, yi1)

                    assert len(xi[yi0[0]]) > yi0[1], (ai, xi, yi0)
                    assert len(xi[yi1[0]]) >= yi1[1], (ai, xi, yi1)
                    w0 = xi[yi0[0]][yi0[1]]
                    w1 = xi[yi1[0]][yi1[1] - 1]
                    ###  Temp fix
                    # if (len(w1) == 0):
                    #     yi1 = (yi1[0],yi1[1]-1)
                    #     w1 = xi[yi1[0]][yi1[1]-1]
                    ### Temp fix end
                    if (ai == article_index):
                        print("\n--------XI[YI0_0][YI0_1]:", xi[yi0[0]][yi0[1]:yi0[1] + 10])
                        print("\n--------XI[YI1]:", xi[yi1[0]][yi1[1] - 10:yi1[1] - 1])
                        print("\n--------W0, W1: ", w0, w1, len(w0), len(w1))
                    i0 = get_word_idx(context, xi, yi0)
                    i1 = get_word_idx(context, xi, (yi1[0], yi1[1] - 1))
                    if (ai == article_index):
                        print("\n--------I0, I1: ", i0, i1)
                    cyi0 = answer_start - i0
                    cyi1 = answer_stop - i1 - 1
                    if (ai == article_index):
                        print("\n--------CYI0, CYI1: ", cyi0, cyi1)
                    # print(answer_text, w0[cyi0:], w1[:cyi1+1])
                    if (ai == article_index):
                        print("\n--------Answer Text[0]", answer_text[0])
                        print("\n--------Answer Text[-1]", answer_text[-1])
                        print("\n--------W0[CYI0]", w0[cyi0])
                        print("\n--------W1[CYI1]: --",)

                    try:
                        assert answer_text[0] == w0[cyi0], (ai, answer_text, w0, cyi0)
                    except:
                        if (w0 not in answer_text):
                            flag = 0

                            # while (flag != 1 and yi0[1] <= yi1[1]):
                            #     yi0 = (yi0[0], yi0[1] + 1)
                            #     w0 = xi[yi0[0]][yi0[1]]
                            #     i0 = get_word_idx(context, xi, yi0)
                            #     cyi0 = answer_start - i0
                            #     print(answer_text, ":", answer_start, i0)
                            #     print(ai, answer_text[0], ":", w0, cyi0)
                            #     print("--------YI0, YI1: ", yi0, yi1)
                            #     print('--')
                            #     if (w0 in answer_text):
                            #         flag = 1
                            #         print("Forward")
                            back_step_count = 0
                            while (flag != 1 and back_step_count < 10):  # yi0[1] <= yi1[1]):
                                back_step_count += 1
                                yi0 = (yi0[0], yi0[1] - 1)
                                w0 = xi[yi0[0]][yi0[1]]
                                i0 = get_word_idx(context, xi, yi0)
                                cyi0 = answer_start - i0
                                # print(answer_text, ":", answer_start, i0)
                                # print(ai, answer_text[0], ":", w0, cyi0)
                                # print("--------YI0, YI1: ", yi0, yi1)
                                # print('--')
                                if (w0 in answer_text):
                                    flag = 1
                            if (flag == 0):
                                # print(answer_text, ":", answer_start, i0)
                                # print(ai, answer_text[0], ":", w0, cyi0)
                                assert_fail_count += 1
                                # print("--------YI0, YI1: ", yi0, yi1)
                                # print("--------BEFORE ASSERT FAIL COUNT: ", assert_fail_count)
                                # print()

                    try:
                        assert answer_text[-1] == w1[cyi1]
                    except:
                        if (w1 not in answer_text):
                            flag = 0
                            while (flag != 1 and yi1[1] >= yi0[1]):
                                yi1 = (yi1[0], yi1[1] - 1)
                                w1 = xi[yi1[0]][yi1[1] - 1]
                                i1 = get_word_idx(context, xi, (yi1[0], yi1[1] - 1))
                                cyi1 = answer_stop - i1 - 1
                                # print(answer_text, ":", answer_stop, i1)
                                # print(ai, answer_text[-1], ":", w1, cyi1)
                                # print("--------YI0, YI1: ", yi0, yi1)
                                # print('--')
                                if (w1 in answer_text):
                                    flag = 1
                                    # if (cyi1 < len(w1) and answer_text[-1] == w1[cyi1]):
                                    #     flag = 1
                            if (flag == 0):
                                # print(answer_text, ":", answer_stop, i1)
                                # print(ai, answer_text[-1], ":", w1, cyi1)
                                assert_fail_count += 1
                                # print("--------YI0, YI1: ", yi0, yi1)
                                # print("-------- AFTER ASSERT FAIL COUNT: ", assert_fail_count)
                                # print()

                    if cyi0 < 32 or cyi1 < 32:
                        len_fail_count += 1
                    # assert cyi0 < 32, (answer_text, w0)
                    # assert cyi1 < 32, (answer_text, w1)

                    yi.append([yi0, yi1])
                    cyi.append([cyi0, cyi1])

                for qij in qi:
                    word_counter[qij] += 1
                    lower_word_counter[qij.lower()] += 1
                    for qijk in qij:
                        char_counter[qijk] += 1

                if len(yi) == 0:
                    print("\n\n\nERROR")
                    print("Not appending\n\n\n")
                else:
                    q.append(qi)
                    cq.append(cqi)
                    y.append(yi)
                    cy.append(cyi)
                    rx.append(rxi)
                    rcx.append(rxi)
                    ids.append(qa['id'])
                    idxs.append(len(idxs))
                    answerss.append(answers)

            if args.debug:
                break

    print("Assert fail count: {}, Len fail count: {}".format(assert_fail_count, len_fail_count))
    word2vec_dict = get_word2vec(args, word_counter)
    lower_word2vec_dict = get_word2vec(args, lower_word_counter)

    # add context here
    data = {'q': q, 'cq': cq, 'y': y, '*x': rx, '*cx': rcx, 'cy': cy,
            'idxs': idxs, 'ids': ids, 'answerss': answerss, '*p': rx}
    shared = {'x': x, 'cx': cx, 'p': p,
              'word_counter': word_counter, 'char_counter': char_counter, 'lower_word_counter': lower_word_counter,
              'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dict}

    print("saving ...")
    save(args, data, shared, out_name)


if __name__ == "__main__":
    main()
