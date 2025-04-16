#assamese characters list
assamese_chars = list(
    "অআইঈউঊঋএঐওঔ"
    "কখগঘঙ"
    "চছজঝঞ"
    "টঠডঢণ"
    "তথদধন"
    "পফবভম"
    "যৰলৱশষসহ"
    "ড়ঢ়য়"
    "ঁংঃ্"
    "০১২৩৪৫৬৭৮৯"
)

#character to index

char_to_idx = {ch: idx + 1 for idx, ch in enumerate(assamese_chars)}
char_to_idx["<blank>"] = 0

#index to character

idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}