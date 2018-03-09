#!/bin/bash

# ten skrypt tworzy diagram klas i pakietow z kodu zrodlowego
# znajdujacego sie w katalogu przypisanym do $sources i zapisuje je
# w katalogu przypisanym do docs

sources="src"
docs="dokumentacja"

pyreverse -s 0 -A -f ALL  --ignore=tests -o png $sources -p diagram
mv packages_diagram.png $docs/packages_diagram.png
mv classes_diagram.png $docs/classes_diagram.png
