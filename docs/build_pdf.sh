#!/usr/bin/env bash
# 由 Markdown 生成 PDF（需 pandoc + xelatex）
cd "$(dirname "$0")"
pandoc 方案说明文档.md -o 方案说明文档.pdf --pdf-engine=xelatex \
  -V CJKmainfont="${CJK_FONT:-Songti SC}" -V mainfont="${MAIN_FONT:-Times New Roman}" -V monofont="${MONO_FONT:-Menlo}" \
  -V geometry:margin=2.2cm -V colorlinks=true --toc
