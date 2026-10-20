#!/usr/bin/env bash
# Завантажує резюме бенчмарку Vanetik & Kogan (2023), якщо їх немає локально (альтернатива `dvc pull`).
set -e
if [ ! -d data/raw/CV ]; then
  git clone --depth 1 https://github.com/NataliaVanetik/vacancy-resume-matching-dataset /tmp/vr
  cp -r /tmp/vr/CV data/raw/CV
fi
echo "data/raw/CV: $(ls data/raw/CV | wc -l) files"
