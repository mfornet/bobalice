import sys
import random
import collections

"""
Greedy Alice is the simplest idea which is smarter than
pure random. She generates a set of random questions,
and stores the responses. Then when asked she finds
the most similar string she saw before, and answers that.
"""

def similarity(w1, w2):
    """
    Computes how similar `w1` and `w2` are using Levenshtein distance.

    Exact strings have 0 similarity
    >>> similarity('abc', 'abc')
    0

    Adding a character gives you 1 similarity
    >>> similarity('abc', 'abcd')
    1

    Inserting a character anywhere gives you 1 similarity
    >>> similarity('abc', 'abbc')
    1

    Swaping a character anywhere gives you 1 similarity
    >>> similarity('abc', 'adc')
    1

    Now a longer test
    >>> similarity('abcd', 'add')
    2
    """
    # Similarity matrix
    distance = collections.defaultdict(lambda: 0)

    for i, c1 in enumerate(w1):
        for j, c2 in enumerate(w2):
            # If c1 == c2 then we have the same distance
            if c1 == c2:
                distance[(i+1, j+1)] = distance[(i,j)]
            else:
                # Otherwise we can past c1 or c2 at either end,
                # or swap them
                keep_c1 = distance[(i, j+1)]
                keep_c2 = distance[(i+1, j)]
                swap = distance[(i, j)]
                distance[(i+1, j+1)] = min(keep_c1, keep_c2, swap) + 1

    return distance[(len(w1), len(w2))]


def main():
    # Read the size of the language
    N = int(raw_input())

    # Read the alfabet
    alphabet = raw_input()

    # Here we store all the previous answers
    memory = []

    # Place the N questions
    for i in range(N):
        # Build a random string of symbols
        length = random.randint(1, 10)
        symbols = [random.choice(alphabet) for k in range(length)]
        question = "".join(symbols)

        # Print and store the response
        print(question)
        sys.stdout.flush()
        answer = raw_input()

        memory.append((question, answer))

    # Asnwer the N responses
    for i in range(N):
        question = raw_input()

        # Find the closest answer and return that
        closest, answer = min(memory, key=lambda t: similarity(t[0], question))

        print(answer)
        sys.stdout.flush()


if __name__ == '__main__':
    main()