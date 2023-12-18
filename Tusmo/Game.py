from Dico import Dico
from Utils import *
from time import sleep

class Game():
    def __init__(self, path:str, region=None):
        self.dico = Dico(path)
        self.not_contain_char = set()
        self.contain_char = set()
        self.good_placed = set()
        self.bad_placed = set()
        self.region = region

    def interpret_result(self, result:str, use_word:str):
        for k, r in enumerate(result):
            if int(r) == -1:
                self.not_contain_char.add(use_word[k])
            elif int(r) == 0:
                self.contain_char.add(use_word[k])
                self.bad_placed.add((k, use_word[k]))
            else:
                self.good_placed.add((k, use_word[k]))

    def check_set(self):
        good_placed = set([c for i,c in self.good_placed] + list(self.contain_char))
        self.not_contain_char = {char for char in self.not_contain_char if char not in good_placed}

    def apply_filter(self):
        self.dico.contain_list_char(self.contain_char)
        self.dico.not_contain_list_char(self.not_contain_char)
        self.dico.list_indice_char(self.good_placed)
        self.dico.list_bad_indice_char(self.bad_placed)

    def play(self):
        if self.region is None:
            self.region = get_grid_region()
            print(self.region)

        text = ""
        if len(text) == 0:
            autoit_script = f"""
                WinActivate(\"TUSMO - Multiplayer Wordle - Google Chrome\")
                Send('{'AB'*7}')
            """
            write_autoit_script(autoit_script, "fill_to_find_length.au3")
            run_autoit_script("fill_to_find_length.au3")
            sleep(1)

        while len(text) == 0:
            text = get_text_from_tentative(self.region, 0)

        length = len(text)
        self.dico.keep_nb_syl(int(length))
        print(f"Lenght found : {length}")

        autoit_script = "Send('{BACKSPACE}')\n"*length
        write_autoit_script(autoit_script, "delete_to_find_length.au3")
        run_autoit_script("delete_to_find_length.au3")
        sleep(1)

        start_char = text[0].lower()
        self.dico.startby(start_char)
        print(f"Start by : {start_char}")

        isFind = False
        tentative = 0

        while (not isFind):
            use_word = self.dico.better_words(min(1, len(self.dico.filter)))[0]
            self.dico.filter.remove(use_word)

            print("\nNext word to use :", use_word)
            print(f"Nb mots possibles : {len(self.dico.filter)}")

            autoit_script = ""
            for char in use_word:
                autoit_script += f"""
                    Send("{char}")
                    Sleep(50)
                """
            autoit_script += "\nSend('{ENTER}')"
            write_autoit_script(autoit_script, "send_word.au3")
            run_autoit_script("send_word.au3")

            sleep(2)

            result = get_word_result(get_tentative(take_screenshot(self.region, True, False), tentative), length)
            print(f"Result : {result}")
            self.interpret_result(result, use_word)
            self.check_set()
            self.apply_filter()

            tentative += 1