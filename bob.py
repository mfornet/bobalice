import sys
import subprocess
import random

from grammar import Language

class Bob:
    """
    Our adversary Bob connects to Alice and handles
    all the gaming.
    """
    def __init__(self, alice, runs, *languages):
        self.alice = alice
        self.languages = list(languages)
        self.rnd = random.Random()
        self.runs = runs

    def run(self):
        runs = 0
        total_accuracy = 0

        results = []

        for l in self.languages:
            for r in range(self.runs):
                accuracy = self._run_once(l)
                total_accuracy += accuracy
                runs += 1
                results.append((l, accuracy))
                print("")

        print("----")

        for l,v in results:
            print("%s ==> %.2f %%" % (l, 100 * v))

        print("----\nTOTAL: %.2f %% accuracy achieved" % (100 * total_accuracy / runs))

    def _run_once(self, language):
        print("Using language: %s" % repr(language))

        alice = subprocess.Popen(self.alice, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        print("Connected to Alice.")

        right = 0
        N = language.size
        alphabet = language.alphabet

        alice.stdin.write('%i\n' % N)
        alice.stdin.write('%s\n' % alphabet)
        alice.stdin.flush()

        for i in range(N):
            question = alice.stdout.readline().strip()
            answer = 'yes' if language.test(question) else 'no'

            print("Got: `%s` ==> `%s`" % (question, answer))
            alice.stdin.write('%s\n' % answer)
            alice.stdin.flush()

        for i in range(N):
            length = self.rnd.randint(1, 10)

            if self.rnd.uniform(0, 1) > 0.5:
                question = "".join(self.rnd.choice(alphabet) for i in range(length))
            else:
                question = language.generate(self.rnd, length)

            expected_answer = 'yes' if language.test(question) else 'no'
            alice.stdin.write('%s\n' % question)
            alice.stdin.flush()

            answer = alice.stdout.readline().strip()

            print("Asked: `%s` ==> `%s` vs `%s`" % (question, answer, expected_answer))

            if answer == expected_answer:
                right += 1

        return right * 1.0 / N


def main():
    bob = Bob(sys.argv[1:], 10,
        Language("Universe", 100, "abc", S="aS bS cS a b c"),
        Language("Even number of A", 100, "ab", S="aO bE b", O="bO aE a", E="aO bE b"),
        Language("Starts with A", 100, "abc", S="aU a", U="aU bU cU a b c"),
        Language("One A and One B", 100, "abc", S="cS aA bB", A="cA bT b", B="cB aT a", T="cT c"),
    )

    bob.run()

if __name__ == '__main__':
    main()
