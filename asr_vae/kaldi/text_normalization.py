import itertools
import re

PUNCTUATION = {
    '"DOUBLE-QUOTE',
    '?QUESTION-MARK',
    '"END-QUOTE',
    ';SEMI-COLON',
    ':COLON',
    '-DASH',
    '(LEFT-PAREN',
    ')RIGHT-PAREN',
    "'SINGLE-QUOTE",
    '!EXCLAMATION-POINT',
    '{LEFT-BRACE',
    '}RIGHT-BRACE',
    '"CLOSE-QUOTE',
    '"END-OF-QUOTE',
    ')CLOSE-PAREN',
    '(BEGIN-PARENS',
    ')END-PARENS',
    ')UN-PARENTHESES',
    ')END-OF-PAREN',
    ')END-THE-PAREN',
    ')CLOSE-BRACE',
    '"IN-QUOTES',
    '(IN-PARENTHESIS',
    ',COMMA',
    '.PERIOD',
    '"DOUBLE-QUOTE',
    '-HYPHEN',
    '.POINT',
    '%PERCENT',
    '--DASH',
    '&ERSAND',
    '&AMPERSAND',
    ':COLON',
    ')RIGHT-PAREN',
    '(LEFT-PAREN',
    ';SEMI-COLON',
    '?QUESTION-MARK',
    '\'SINGLE-QUOTE',
    '...ELLIPSIS',
    '/SLASH',
    '}RIGHT-BRACE',
    '{LEFT-BRACE',
    '!EXCLAMATION-POINT',
    '+PLUS',
    '=EQUALS',
    '#SHARP-SIGN',
    '-MINUS',
    '"QUOTE',
    '"UNQUOTE',
    '(PAREN',
    '(PARENTHESES',
    ')PAREN',
    '(PARENTHETICALLY',
    ')CLOSE_PAREN',
    '(BRACE'

}


def normalize_punctuation(p):
    words = re.split(pattern="[\\-_]", string=p)
    words = (re.sub(pattern="\\W+", repl="", string=w) for w in words)
    words = (w for w in words if len(w) > 0)
    return words


PUNCTUATION_SUBS = {
    k: normalize_punctuation(k) for k in PUNCTUATION
}


def normalize_word_for_training(word):
    w = word
    if re.match(
            pattern="^<.*>$",
            string=w):
        return []
    if re.match(
            pattern="^~+$",
            string=w):
        return []
    # if w[0] == "*" and w[-1] == "*":
    #    w = w[1:-1]
    w = re.sub(
        pattern="\\*",
        repl="",
        string=w
    )
    w = re.sub(
        pattern="`",
        repl="'",
        string=w
    )
    if w in PUNCTUATION_SUBS:
        return PUNCTUATION_SUBS[w]
    w = re.sub(
        pattern="!",
        repl="",
        string=w
    )
    w = re.sub(
        pattern=";",
        repl="",
        string=w
    )
    # if w[0] == "!":
    #    w = w[1:]
    w = re.sub(
        pattern="\\(.*\\)",
        repl="",
        string=w
    )
    w = re.sub(
        pattern=":",
        repl="",
        string=w
    )
    if w[0] == "-":
        w = w[1:]
    if w[-1] == "-":
        w = w[:-1]
    # ws = w.split("-")
    return [w]


def normalize_sentence_for_training(sentence):
    words = sentence.split(" ")
    words = (normalize_word_for_training(word) for word in words)
    words = itertools.chain.from_iterable(words)
    # words = (word for word in words if word is not None)
    return " ".join(words)


if __name__ == '__main__':
    print(normalize_sentence_for_training(
        "DRAMS (IN-PARENTHESIS DYNAMIC RANDOM ACCESS MEMORIES ARE A PARTICULAR EXAMPLE"
    ))
    print(PUNCTUATION_SUBS)
