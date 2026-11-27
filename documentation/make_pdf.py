#====================================================================
# Makes documentation.pdf from documentation.md. Run it again after
# changing documentation.md:  python documentation/make_pdf.py
#
# Needs pandoc, which is included in the Python package
# pypandoc_binary (pip install pypandoc_binary), and a LaTeX
# installation with pdflatex (e.g. TeX Live or MiKTeX).
#====================================================================
import os
import re
import tempfile

import pypandoc

#Line breaks in long code lines, smaller tables, and the box characters in the folder tree
LATEX_HEADER = r"""
\usepackage{pmboxdraw}
\usepackage{fvextra}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,fontsize=\small,commandchars=\\\{\}}
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\small}
"""

if __name__ == "__main__":
    folder = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as temporary_folder:
        header_file = os.path.join(temporary_folder, "header.tex")
        with open(header_file, "w", encoding="utf-8") as f:
            f.write(LATEX_HEADER)
        with open(os.path.join(folder, "documentation.md"), encoding="utf-8") as f:
            text = f.read()
        #GitHub writes equations in ```math blocks, while pandoc uses $$ ... $$
        text = re.sub(r"```math\n(.*?)\n```", r"$$\n\1\n$$", text, flags=re.S)
        pypandoc.convert_text(
            text,
            "pdf",
            format="markdown",
            outputfile=os.path.join(folder, "documentation.pdf"),
            extra_args=[
                "--pdf-engine=pdflatex",
                "--shift-heading-level-by=-1",  #The first heading becomes the title
                f"--include-in-header={header_file}",
                "--variable=papersize:a4",
                "--variable=geometry:margin=2.2cm",
                "--variable=fontsize:11pt",
                "--variable=colorlinks:true",
            ])
    print(f"Made {os.path.join(folder, 'documentation.pdf')}")
