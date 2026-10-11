#!/usr/bin/env bash
# 由 Markdown 生成 PDF（需 pandoc + xelatex）：方案说明文档（带目录）与作品介绍
cd "$(dirname "$0")"
FONTS=(-V CJKmainfont="${CJK_FONT:-Songti SC}" -V mainfont="${MAIN_FONT:-Times New Roman}" -V monofont="${MONO_FONT:-Menlo}")
pandoc 方案说明文档.md -o 方案说明文档.pdf --pdf-engine=xelatex "${FONTS[@]}" -V geometry:margin=2.2cm -V colorlinks=true --toc
pandoc 作品介绍.md -o 作品介绍.pdf --pdf-engine=xelatex "${FONTS[@]}" -V geometry:margin=2.0cm -V colorlinks=true
