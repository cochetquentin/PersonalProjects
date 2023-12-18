import numpy as np

def supprime_accent(ligne):
    """ supprime les accents du texte source """
    accent = ['é', 'è', 'ê', 'à', 'ù', 'û', 'ç', 'ô', 'î', 'ï', 'â', "ü"]
    sans_accent = ['e', 'e', 'e', 'a', 'u', 'u', 'c', 'o', 'i', 'i', 'a', "u"]
    i = 0
    while i < len(accent):
        ligne = ligne.replace(accent[i], sans_accent[i])
        i += 1
    return ligne

class Dico():
    def __init__(self, path:str, encoding="utf-8") -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.mots = f.readlines()

        self.mots = [supprime_accent(m.rstrip()) for m in self.mots if m.find("-") == -1]
        self.mots = list(set(self.mots))
        self.filter = self.mots

    def keep_nb_syl(self, length : int) -> None:
        self.filter = [m for m in self.mots if len(m)==length]

    def contain_char(self, char : str) -> None:
        self.filter = [m for m in self.filter if m.find(char) != -1]

    def contain_list_char(self, l:list) -> None :
        for mot in l:
            self.contain_char(mot)

    def not_contain_char(self, char : str) -> None:
        self.filter = [m for m in self.filter if m.find(char) == -1]

    def not_contain_list_char(self, l:list) -> None :
        for mot in l:
            self.not_contain_char(mot)

    def startby(self, char:str) -> None:
        self.filter = [m for m in self.filter if m.startswith(char)]

    def indice_char(self, indice:int, char:str) -> None:
        self.filter = [m for m in self.filter if m[indice] == char]

    def list_indice_char(self, liste:list) -> None:
        for i, c in liste:
            self.indice_char(i,c)

    def better_words(self, nb:int) -> list :
        len_words = [len(set(w)) for w in self.filter]
        idx_max = np.argpartition(len_words, len(len_words)-nb)[-nb:]
        return np.array(self.filter)[idx_max]

    def bad_indice_char(self, indice:int, char:str) -> None:
        self.filter = [m for m in self.filter if m[indice] != char]

    def list_bad_indice_char(self, liste:list) -> None:
        for i, c in liste:
            self.bad_indice_char(i,c)