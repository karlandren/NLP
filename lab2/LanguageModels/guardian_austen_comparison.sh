#!/bin/sh

# guardian on guardian 6.72 (real 6.62)
# guardian on austen 6.47 (real 6.40)
python3 BigramTester.py -f guardian_model.txt -t data/guardian_test.txt --check
python3 BigramTester.py -f guardian_model.txt -t data/austen_test.txt --check

# austen on austen 5.80 (real 5.72)
# austen on guardian 9.80 (9.75)
python3 BigramTester.py -f austen_model.txt -t data/austen_test.txt --check
python3 BigramTester.py -f austen_model.txt -t data/guardian_test.txt --check

# - many of the words in austen_train are unlikely to appear in guardian.
# - guardian_model contains more unique words than austen_model (26 times as many)
# - guardian_test is much larger than austen_test, which also could explain why guardian
# gets 6.47 on austen but austen gets 9.8 on guardian