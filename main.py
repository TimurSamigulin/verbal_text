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


if __name__ == '__main__':
    sent = "Привет, убить тебя мало, человечишка ты"
    print(get_imperative(sent))
    print(get_small_talk(sent))
    print(get_mat(sent))
    print(get_insult(sent))
    print(get_threat(sent))
