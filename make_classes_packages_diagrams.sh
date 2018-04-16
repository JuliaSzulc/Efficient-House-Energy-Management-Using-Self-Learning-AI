#!/bin/bash

# ten skrypt tworzy diagram klas i pakietow z kodu zrodlowego
# znajdujacego sie w katalogu przypisanym do $sources i zapisuje je
# w katalogu przypisanym do docs

# uwaga, ten program pyreverse nie jest idealny - ten diagram klas to jest
# tak w 80% gotowy, bo Python często nie jest w stanie stwierdzić jakiego typu
# będzie dana zmienna i nie widzi przez to powiązań itd. Diagram w finalnej
# wersji wymaga jeszcze obróbki.

sources="src"
docs="dokumentacja"

pyreverse -s 0 -f ALL  --ignore=tests -o png $sources -p diagram
mv packages_diagram.png $docs/packages_diagram.png
mv classes_diagram.png $docs/classes_diagram.png
