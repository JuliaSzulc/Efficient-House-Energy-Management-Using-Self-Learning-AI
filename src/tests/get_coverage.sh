#!/bin/bash
coverage erase
for file in $(ls | grep -v '.*~\|__.*' | grep ".*_tests.py"); do
    echo "running" $file "with coverage.py..."
    coverage run -a --source .. --omit */tests/* $file
done
echo "Coverage report from tests:"
coverage report -m
echo "Opening html file..."
coverage html
xdg-open htmlcov/index.html

