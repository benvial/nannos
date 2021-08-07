#!/usr/bin/env python
"""
%prog MODE FILES...

Post-processes HTML and Latex files output by Sphinx.
MODE is either 'html' or 'tex'.

"""
from __future__ import absolute_import, division, print_function

import io
import optparse
import re

import nannos as package


def main():
    p = optparse.OptionParser(__doc__)
    options, args = p.parse_args()

    if len(args) < 1:
        p.error("no mode given")

    mode = args.pop(0)

    if mode not in ("html", "tex"):
        p.error("unknown mode %s" % mode)

    for fn in args:
        f = io.open(fn, "r", encoding="utf-8")
        try:
            if mode == "html":
                lines = process_html(fn, f.readlines())
            elif mode == "tex":
                lines = process_tex(f.readlines())
        finally:
            f.close()

        f = io.open(fn, "w", encoding="utf-8")
        f.write("".join(lines))
        f.close()


def process_html(fn, lines):

    new_lines = []
    for line in lines:

        # Remove escaped arguments from the html files.
        line = line.replace("\*args", "*args")
        line = line.replace("\*\*kwargs", "**kwargs")
        line = line.replace("col-md-3", "col-md-2")
        line = line.replace(
            "<title> &#8212;",
            f"<title> {package.__name__}: {package.__description__} &#8212;",
        )

        line = line.replace(". URL: ", ".")
        if line.startswith("<dd><p>") and line.endswith("</a>.</p>\n"):
            line = line.replace("</a>.</p>", "</a></p>")

        line = line.replace(
            "https://mybinder.org/v2/gh/nannos/nannos",
            "https://mybinder.org/v2/gl/nannos%2Fnann",
        )

        line = line.replace(
            "https://mybinder.org/v2/gl/nannos%2Fnann/doc?filepath=notebooks/examples/",
            "https://mybinder.org/v2/gl/nannos%2Fnann/doc?filepath=notebooks/",
        )
        line = line.replace("binder_badge_logo1.svg", "binder_badge_logo.svg")
        line = line.replace("binder_badge_logo2.svg", "binder_badge_logo.svg")

        line = line.replace(
            'binder" src="../../_images/binder_badge_logo.svg" width="150px"',
            'binder" src="../../_images/binder_badge_logo.svg" width="200px"',
        )
        line = line.replace(
            'binder" src="../_images/binder_badge_logo.svg" width="150px"',
            'binder" src="../_images/binder_badge_logo.svg" width="200px"',
        )

        new_binder_badge = "https://img.shields.io/badge/launch-binder-d7a44c.svg?style=for-the-badge&logo=data:image/png;base64,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"
        line = line.replace(
            "../../_images/binder_badge_logo.svg",
            new_binder_badge,
        )
        line = line.replace(
            "../_images/binder_badge_logo.svg",
            new_binder_badge,
        )
        # line = re.sub('<code class="xref download docutils literal notranslate"><span class="pre">Download</span> <span class="pre">Python</span> <span class="pre">source</span> <span class="pre">code(.*?)</span> <span class="pre">plot_scattering2d_pec_cylinder.py</span></code></a></p>$',"",line,flags=re.DOTALL)

        icon_python = '<i class="icondld icon-python"></i>'
        icon_jupyter = '<i class="icondld icon-jupyter"></i>'

        first_tag = '<span class="pre">Download</span> <span class="pre">Python</span> <span class="pre">source</span> <span class="pre">code'
        second_tag = "</span></code></a></p>"
        reg = "(?<=%s).*?(?=%s)" % (first_tag, second_tag)
        line = re.sub(reg, "", line, flags=re.DOTALL)
        if icon_python not in line:
            line = line.replace(first_tag, icon_python + first_tag)

        first_tag = '<span class="pre">Download</span> <span class="pre">Jupyter</span> <span class="pre">notebook'
        reg = "(?<=%s).*?(?=%s)" % (first_tag, second_tag)
        line = re.sub(reg, "", line, flags=re.DOTALL)
        if icon_jupyter not in line:
            line = line.replace(first_tag, icon_jupyter + first_tag)

        first_tag = '<span class="pre">Download</span> <span class="pre">all</span> <span class="pre">examples</span> <span class="pre">in</span> <span class="pre">Python</span> <span class="pre">source</span> <span class="pre">code'
        second_tag = "</span></code></a></p>"
        reg = "(?<=%s).*?(?=%s)" % (first_tag, second_tag)
        line = re.sub(reg, "", line, flags=re.DOTALL)
        if icon_python not in line:
            line = line.replace(first_tag, icon_python + first_tag)

        first_tag = '<span class="pre">Download</span> <span class="pre">all</span> <span class="pre">examples</span> <span class="pre">in</span> <span class="pre">Jupyter</span> <span class="pre">notebooks'
        reg = "(?<=%s).*?(?=%s)" % (first_tag, second_tag)
        line = re.sub(reg, "", line, flags=re.DOTALL)
        if icon_jupyter not in line:
            line = line.replace(first_tag, icon_jupyter + first_tag)

        new_lines.append(line)
    return new_lines


def process_tex(lines):
    """
    Remove unnecessary section titles from the LaTeX file.

    """
    new_lines = []
    for line in lines:
        if (
            line.startswith(r"\section{nannos.")
            or line.startswith(r"\subsection{nannos.")
            or line.startswith(r"\subsubsection{nannos.")
            or line.startswith(r"\paragraph{nannos.")
            or line.startswith(r"\subparagraph{nannos.")
        ):
            pass  # skip!
        else:
            new_lines.append(line)
    return new_lines


if __name__ == "__main__":
    main()
