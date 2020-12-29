import unicodedata

path_to_file = "./por-eng/por.txt"


# Converte o arquivo Unicode para ASCII
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')