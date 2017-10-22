import sys
import random

"""
Random Alice simply answers random to everything.
It should serve as a benchmark, i.e., every other
implementation should be better than her.
"""

def main():
    # Read the size of the language
    N = int(raw_input())

    # Read the alfabet
    alphabet = raw_input()

    # Place the N questions
    for i in range(N):
        # Build a random string of symbols
        length = random.randint(1, 10)
        symbols = [random.choice(alphabet) for k in range(length)]
        question = "".join(symbols)

        # Print and throw away the response
        print(question)
        sys.stdout.flush()

        answer = raw_input()

        # But at least check is OK
        assert answer in ['yes', 'no']

    # Asnwer the N responses
    for i in range(N):
        question = raw_input()

        # Is Bob cheating ?
        assert all(symbol in alphabet for symbol in question)

        answer = random.choice(['yes', 'no'])
        print(answer)
        sys.stdout.flush()


if __name__ == '__main__':
    main()
