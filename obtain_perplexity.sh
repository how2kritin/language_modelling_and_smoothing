#!/bin/bash

N=(1 3 5)
lm_type=("l" "g" "i")
corpora=("corpus/Pride and Prejudice - Jane Austen.txt" "corpus/Ulysses - James Joyce.txt")

for corpus in "${corpora[@]}"; do
  for n in "${N[@]}"; do
    for type in "${lm_type[@]}"; do
      echo "Running $n $type $corpus"
      python ./src/language_model.py "$n" "$type" "$corpus" pe
    done
  done
done
