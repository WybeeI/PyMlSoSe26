"""
The classic interview hazing ritual, now with extra chaos.

Rules:
  - For multiples of 3:           print "Foo"
  - For multiples of 5:           print "Bar"
  - For multiples of 7:           print "Bop"
  - For multiples of 3 AND 5:     print "FooBar"
  - For multiples of 3 AND 7:     print "FooBop"
  - For multiples of 5 AND 7:     print "BarBop"
  - For multiples of 3, 5, AND 7: print "FooBarBop"
  - For multiples of 11:          append "Bap" to whatever you would print
                                  (or print "Bap" alone if no other rule fires)
  - For everything else:          print the number itself

Example: foobar(1, 16) should print:
    1
    2
    Foo
    4
    Bar
    Foo
    Bop
    8
    Foo
    Bar
    Bap
    Foo
    13
    14
    FooBar

DO NOT MODIFY THE FUNCTION SIGNATURES.
"""


def foobar_word(n: int) -> str:
    word = ""

    # Reglas base
    if n % 3 == 0:
        word += "Foo"
    if n % 5 == 0:
        word += "Bar"
    if n % 7 == 0:
        word += "Bop"

    # Regla especial de 11
    if n % 11 == 0:
        if word == "":
            word = "Bap"
        else:
            word += "Bap"

    # Si no se generó nada, devolver el número
    if word == "":
        return str(n)

    return word


def foobar(start: int, stop: int) -> None:
    for n in range(start, stop):
        print(foobar_word(n))

# ---------------------------------------------------------------------------
# Prompt / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== FooBarBop(Bap) ===")
    print("Rules: Foo=÷3  Bar=÷5  Bop=÷7  Bap=÷11  (combined for multiple divisors)\n")

    try:
        start = int(input("Start (inclusive): "))
        stop = int(input("Stop  (exclusive): "))
    except ValueError:
        print("Please enter integers.")
        raise SystemExit(1)

    print()
    foobar(start, stop)
