# char_map.py
assamese_chars = list(
    "অআইঈউঊঋএঐওঔ"  # Vowels
    "কখগঘঙচছজঝঞ"      # Consonants
    "টঠডঢণতথদধন"
    "পফবভমযৰলৱ"
    "শষসহড়ঢ়য়"      # Includes conjunct forms (ra+nukta, dha+nukta, ya+nukta)
    "ঁংঃ্"             # Diacritics
    "ািীুূৃেৈোৌ"      # Vowel signs
    "০১২৩৪৫৬৭৮৯"      # Assamese numerals
    "ক্কক্তক্মক্ষগ্নগ্মঙ্গঙ্কচ্চজ্জঞ্চঞ্জট্টড্ডন্তন্দন্নপ্তপ্পব্দব্ধম্পম্বম্মল্লশ্চশ্মষ্টস্তস্থস্পস্মহ্নহ্মন্ত্রস্ত্রক্ষ্ম"  # Conjuncts
    "।,"               # Punctuation
    " "                # Space
    "\t"               # Tab
    "!\"'()*./:;?_"    # Additional punctuation
    "-"                # Hyphen
    "0123456789"       # Arabic numerals
    "ABDIKMWabcdefghijlmnoprstuvwy"  # English letters
    "ৎ৷॥"             # Assamese-specific (khando-ta, alt full stop, double danda)
    "‌"                # Zero-width joiner
    "ড়ঢ়য়"             # Standalone Assamese characters (ra with dot, rha with dot, ya-phala)
)

char_to_idx = {char: idx + 1 for idx, char in enumerate(assamese_chars)}  # 1-based for blank
idx_to_char = {idx: char for char, idx in char_to_idx.items()}