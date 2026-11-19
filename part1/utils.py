import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation
    
    # extract the text, we also lower everything so our transformations are case insensitive
    # this also mimics casual typing where people dont capitalize
    text = example["text"].lower()

    #1. SLANG
    # I asked my own AI - GLM 5.3 to generate a list of regular words and their slang equivalents.
    # I then randomly selected each word with 0.75 probability to replace by a slang equivalent.
    # keys are all lowercase because the text is lowercase. 
    SLANG = {
    # positive
    "very good": "fire",
    "really good": "fire",
    "amazing": "insane",
    "incredible": "unreal",
    "awesome": "goated",
    "fantastic": "goated",
    "excellent": "elite",
    "great": "peak",
    "wonderful": "elite",
    "masterpiece": "peak",
    "beautiful": "gorge",
    "well acted": "so clean",
    "i loved it": "it slapped",
    "i loved this movie": "this movie ate",
    "i really enjoyed": "i really messed with",
    "enjoyed it": "messed with it",
    "the acting was great": "the acting ate",
    "i was blown away": "i was not ready",
    "highly recommend": "do not sleep on this",
    "the best movie": "a certified banger",
    "love": "fw",
    "loved": "was obsessed with",
    "perfect": "goated",

    # negative
    "very bad": "actual dogwater",
    "bad": "mid",
    "terrible": "trash",
    "awful": "tragic",
    "horrible": "hot garbage",
    "boring": "a snoozefest",
    "the worst": "bottom tier",
    "waste of time": "hard skip",
    "poorly written": "cooked",
    "the plot was slow": "the plot was dragging",
    "disappointing": "washed",
    "i hated it": "i want these hours back",
    "i couldn't finish it": "i tapped out early",
    "hate": "can't rock with",
    "hated": "couldn't rock with",

    # neutral flavor, optional
    "definitely": "deadass",
    "really": "legit",
    }

    # probability of replacement with slang
    p_slang = 0.75

    # slang pass: we want to check longest phrases first, so phrases such as "very good" win over simple phrases such as "good"
    for phrase in sorted(SLANG, key=len, reverse=True):
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b")

        def swap_or_keep(m):
            # if random decimal is less than our probability, we replace with slang
            if random.random() < p_slang:
                return SLANG[phrase]
            # otherwise we keep the original text
            return m.group(0)

        # we replace the text with slang or keep it based on our probability
        text = pattern.sub(swap_or_keep, text)

    # replace the example text with the new text
    example["text"] = text

    #2. Sarcasm
    # Sarcasm is also very interesting and can happen in reviews.
    # It is also hard, very hard to implement manually, so I asked my own AI - GLM 5.3 to generate a list of sarcastic phrases that can be used in reviews.
    # We are going to append these phrases to the end of the text with 1 100% probability depending on the label.
    # we lowercase the phrases when appending so they match the rest of the review

    # sarcasm phrases for positive label
    sarcasm_positive = [
    # mock-complaints that decodes positive
    "Oh no, a movie that's actually good, however will I cope.",
    "Ugh, competent acting AND an actual plot? Disgusting. Loved every second.",
    "Please, no more well-written characters, I can't take it.",
    "Devastating news: the movie was too good and now everything else looks bad.",
    "I was SO angry when it ended. That's how you know it worked.",
    "Oh sure, rub it in, not every film gets to be this decent.",
    ]

    sarcasm_negative = [
    # mock-praise that decodes negative
    "Oh sure, TOTALLY the highlight of my year. I just LOVE wasting two hours.",
    "Yeah, ten out of ten, no notes, I adore films where absolutely nothing happens.",
    "A masterpiece. A triumph. I was on the edge of my seat... checking my phone.",
    "Wow, incredible, they really invented boredom all over again.",
    "Would happily pay full price again to relearn what 'plot' means.",
    "Gripping stuff, if you find paint drying gripping.",
    ]

    # probability for sarcasm
    p_sarcasm = 1.0

    # now we append sarcasm sentences if random decimal is less than our probability, and we select the sarcasm sentence based on the label
    if random.random() < p_sarcasm:
        pool = sarcasm_negative if example["label"] == 0 else sarcasm_positive
        text = text + " " + random.choice(pool).lower()

    # now replace the example text with the new text
    example["text"] = text

    #3. Emoji
    # Emoji are also very fun and relevant way to write reviews.
    # Emojis can actually be used to convey sentiment and emotion, so we can use them.

    # To avoid destroying the meaning of the text, we will only append emojis to the end of the text, and we will only append them with 0.5 probability.
    # Since doing emojis is a little complicated we will only use few trivial emojis for this assignment.
    # Good emojis such as fire and heart can be used for positive sentiment, while trash and eye roll can be used for negative sentiment.
    EMOJI = {
    1: ["🔥", "❤️"],
    0: ["🗑️", "🙄"],
    }
    # we again make probability to 0.5
    p_emoji = 0.5

    # emoji list for our label
    label = EMOJI[example["label"]]

    # if random decimal is less than our probability, we will append an emoji to the end of the text.
    if random.random() < p_emoji:
        text = text + " " + random.choice(EMOJI[example["label"]])

    # now replace the example text with the new text
    example["text"] = text

    #4. Typos
    # I did not want to use typos at first because it is one of the provided examples
    # and I wanted my transformation to be original. But my own three parts could not
    # reach the 4 point drop that is needed for the full grade, so I add typos as a
    # fourth part. It also fits the theme, since people type fast and make typos
    # when they write reviews on their phone.
    # We swap one letter in some words for a neighbor key on the keyboard,
    # which is the most common typo people make.
    # keys of the map are letters, values are the neighbor letters on a qwerty keyboard
    NEIGHBORS = {
    "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "ryfg", "y": "tugh", "u": "yihj", "i": "uojk", "o": "ipkl", "p": "ol",
    "a": "qwsz", "s": "aedxzw", "d": "serfcx", "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "j": "huikmn", "k": "jiolm", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
    }
    # probability that a word gets a typo
    p_typo = 0.15

    # find all lowercase words with 4 or more letters,
    # and for some of them swap one letter for a neighbor key
    def make_typo(m):
        word = m.group(0)
        # if random decimal is less than our probability, we make a typo in this word
        if random.random() < p_typo:
            # pick a random position in the word
            i = random.randrange(len(word))
            # pick a random neighbor of that letter
            choices = NEIGHBORS[word[i]]
            # we make sure we actually change the letter
            if len(choices) > 0:
                word = word[:i] + random.choice(choices) + word[i+1:]
        return word

    text = re.sub(r"\b[a-z]{4,}\b", make_typo, text)

    # now replace the example text with the final text
    example["text"] = text

    ##### YOUR CODE ENDS HERE ######

    return example
