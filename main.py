from app.check_insult import BertPredict
from app.mat_filter import count_mat_detect
from app.morph import Imperative
from app.small_talk import SmallTalk


def get_imperative(sent: str) -> bool:
    """
        Строка на вход, возвращает True если есть повелительное наклонение
        param:
            sent: str - предложение
        return:
            bool - наличие повелительного наклонения в предложение
    """
    imper = Imperative()
    return imper.isPovel(sent)


def get_small_talk(sent: str) -> tuple:
    """
        Проверка на уменьшительно-ласкательные слова
        param:
            sent: str - предложение
        return:
            tuple(bool, list) - (наличие этих слова, множество уменьшительно-ласкательных слова)
    """
    small = SmallTalk()
    affect, words_affect = small.isAffect(sent)
    return affect, words_affect


def get_mat(sent: str) -> tuple:
    """
    Подсчёт матерных слов

    :param text: Текст который нужно проанализировать задаётся как str. Может быть любой длины
    :type text: str
    :return: Возвращает кортеж. В первой ячейке которого содержится количество матерных слов, во второй - процент матерных слов в текста и в третьей - множество матерных слов.
    :rtype: tuple
    """
    return count_mat_detect(sent)


def get_insult(sent: str) -> bool:
    """
        Проверка на наличие оскорблений
        param:
            sent: str - предложение
        return:
            bool - True если в тексте есть оскорбления 
    """
    insult = BertPredict(model_path='app/data/insult_bert')
    return insult.predict(sent)


def get_threat(sent: str) -> bool:
    """
        Проверка на наличие угроз
        param:
            sent: str - предложение
        return:
            bool - True если в тексте есть угрозы 
    """
    threat = BertPredict(model_path='app/data/threat_bert')
    return threat.predict(sent)


def get_toxic(sent: str) -> bool:
    """
        Проверка на токсичность
        param:
            sent: str - предложение
        return:
            bool - True если в текст токсичные
    """
    toxic = BertPredict(model_path='app/data/rubert-toxic-detection')
    return toxic.predict(sent)


if __name__ == '__main__':
    print(get_toxic('Ты очень красивая'))
    # import json
    # import re
    # import pandas as pd

    # with open("data/stt.json", 'r') as fr:
    #     texts = json.load(fr)

    # res_texts = []

    # for key, text in texts.items():
    #     print(key)
    #     sents = re.split("\.|\!|\?", text)
    #     for sent in sents:
    #         if not sent:
    #             continue
    #         sent_res = {}
    #         sent_res['sent'] = sent
    #         sent_res['is_imperative'] = get_imperative(sent)

    #         small_talk = get_small_talk(sent)
    #         sent_res['is_affect'] = small_talk[0]
    #         sent_res['words_affect'] = small_talk[1]

    #         mat = get_mat(sent)
    #         sent_res['count_mat'] = mat[0]
    #         sent_res['prop_mat'] = mat[1]
    #         sent_res['list_mat'] = mat[2]

    #         sent_res['is_insult'] = get_insult(sent)
    #         sent_res['is_threat'] = get_threat(sent)
    #         sent_res['text'] = text
    #         sent_res['key'] = key

    #         res_texts.append(sent_res)

    # df = pd.DataFrame(res_texts)
    # df.to_csv('data/result_agg.csv', header=True, index=True, sep=';')
