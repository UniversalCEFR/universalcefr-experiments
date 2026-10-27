"""Language packages and morphological categories used by the original extractor."""

lang_dict = {
    "ar": "padt", "cs": "pdt", "cy": "ccg", "de": "hdt", "en": "gum",
    "es": "ancora", "et": "edt", "fr": "sequoia", "hi": "hdtb", "it": "isdt",
    "nl": "alpino", "pt": "bosque", "ru": "syntagrus",
}

mood_dict = {"Cnd": 0, "Imp": 0, "Ind": 0, "Jus": 0, "Qot": 0, "Sub": 0}
tense_dict = {"Fut": 0, "Imp": 0, "Past": 0, "Pqp": 0, "Pres": 0}
voice_dict = {"Act": 0, "Pass": 0, "Mid": 0}
polarity_dict = {"Pos": 0, "Neg": 0}
case_dict = {
    "Nom": 0, "Gen": 0, "Dat": 0, "Acc": 0, "Erg": 0, "Voc": 0, "Ins": 0,
    "Ess": 0, "Tra": 0, "Abe": 0, "Ben": 0, "Cau": 0, "Cmp": 0, "Cns": 0,
    "Com": 0, "Dis": 0, "Equ": 0, "Par": 0, "Tem": 0,
}
number_dict = {"Dual": 0, "Sing": 0, "Plur": 0}
foreign_dict = {"Yes": 0}
abbr_dict = {"Yes": 0}
definite_dict = {"Cons": 0, "Com": 0, "Def": 0, "Ind": 0}

morph_dict_verbs = {
    "Mood": mood_dict, "Tense": tense_dict, "Voice": voice_dict,
    "Polarity": polarity_dict,
}
morph_dict_nouns = {
    "Case": case_dict, "Number": number_dict, "Foreign": foreign_dict,
    "Abbr": abbr_dict, "Definite": definite_dict,
}
