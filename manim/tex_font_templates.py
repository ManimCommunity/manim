import os
from .utils.tex import TexTemplateFromFile
from .mobject.svg.tex_mobject import TextMobject
from .utils.config_ops import digest_config


class TexTemplateFromFontConfig(TexTemplateFromFile):
    def __init__(self, font, **kwargs):
        digest_config(self, kwargs)
        self.font=font
        if "compiler" in TEX_FONT_CONFIGS[font]:
            tex_command = TEX_FONT_CONFIGS[font]["compiler"]
        else:
            tex_command = "latex"
        if "output" in TEX_FONT_CONFIGS[font]:
            tex_output = TEX_FONT_CONFIGS[font]["output"]
        else:
            tex_output = ".dvi"
        self.tex_compiler = {
            "command": tex_command,
            "output_format": tex_output,
        }
        self.rebuild_cache()

    def rebuild_cache(self):
        self.body = self.TexTemplateFileBodyFromFontConfig(self.font)

    def TexTemplateFileBodyFromFontConfig(self, font_name):
        template = open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "tex_font_template.tex",
                ),
            "r").read()
        file_body = \
            template.replace(
                "FontConfig",
                TEX_FONT_CONFIGS[font_name]["config"] + (
                    "" if r"\begin{document}" in TEX_FONT_CONFIGS[font_name]["config"] else
                    r"""
                    \begin{document}
                    """)
            )
        return file_body

class FontMobject(TextMobject):
    def __init__(self, *tex_strings, font=None, **kwargs):
        if font is None: # default to TextMobject
            super().__init__(*tex_strings, **kwargs)
        else:
            super().__init__(*tex_strings, template=TexTemplateFromFontConfig(font), **kwargs)

# All the font examples from http://jf.burnol.free.fr/showcase.html

TEX_FONT_CONFIGS = {
    "lmtp" : {
        "description": "Latin Modern Typewriter Proportional",
        "config": r"""
            \usepackage[T1]{fontenc}
            \usepackage[variablett]{lmodern}
            \renewcommand{\rmdefault}{\ttdefault}
            \usepackage[LGRgreek]{mathastext}
            \MTgreekfont{lmtt} % no lgr lmvtt, so use lgr lmtt
            \Mathastext
            \let\varepsilon\epsilon % only \varsigma in LGR
            """,
        },
    "fufug": {
        "description": "Fourier Utopia (Fourier upright Greek)",
        "config": r"""
            \usepackage[T1]{fontenc}
            \usepackage[upright]{fourier}
            \usepackage{mathastext}
            """,
        },
    "droidserif": {
        "description": "Droid Serif",
        "config": r"""
            \usepackage[T1]{fontenc}
            \usepackage[default]{droidserif}
            \usepackage[LGRgreek]{mathastext}
            \let\varepsilon\epsilon
            """,
        },
    "droidsans": {
        "description": "Droid Sans",
        "config": r"""
            \usepackage[T1]{fontenc}
            \usepackage[default]{droidsans}
            \usepackage[LGRgreek]{mathastext}
            \let\varepsilon\epsilon
            """,
        },
    "ncssg": {
        "description": "New Century Schoolbook (Symbol Greek)",
        "config": r"""
            \usepackage[T1]{fontenc}
            \usepackage{newcent}
            \usepackage[symbolgreek]{mathastext}
            \linespread{1.1}
            """,
        },
    "fceg": {
        "description": "French Cursive (Euler Greek)",
        "config": r"""
            \usepackage[T1]{fontenc}
            \usepackage[default]{frcursive}
            \usepackage[eulergreek,noplusnominus,noequal,nohbar,%
            nolessnomore,noasterisk]{mathastext}
            """,
        },
    "aksg": {
        "description": "Auriocus Kalligraphicus (Symbol Greek)",
        "config": r"""
            \usepackage[T1]{fontenc}
            \usepackage{aurical}
            \renewcommand{\rmdefault}{AuriocusKalligraphicus}
            \usepackage[symbolgreek]{mathastext}
            """,
        },
    "ecfscmg": {
        "description": "ECF Skeetch (CM Greek)",
        "config": r"""
            \usepackage[T1]{fontenc}
            \usepackage[T1]{fontenc}
            \DeclareFontFamily{T1}{fsk}{}
            \DeclareFontShape{T1}{fsk}{m}{n}{<->s*[1.315] fskmw8t}{}
            \renewcommand\rmdefault{fsk}
            \usepackage[noendash,defaultmathsizes,nohbar,defaultimath]{mathastext}
            """,
        },
    "urwagsg": {
        "description": "URW Avant Garde (Symbol Greek)",
        "config": r"""
            \usepackage[T1]{fontenc}
            \usepackage{avant}
            \renewcommand{\familydefault}{\sfdefault}
            \usepackage[symbolgreek,defaultmathsizes]{mathastext}
            """,
        },
    "urwzccmg": {
        "description": "URW Zapf Chancery (CM Greek)",
        "config": r"""
            \usepackage[T1]{fontenc}
            \DeclareFontFamily{T1}{pzc}{}
            \DeclareFontShape{T1}{pzc}{mb}{it}{<->s*[1.2] pzcmi8t}{}
            \DeclareFontShape{T1}{pzc}{m}{it}{<->ssub * pzc/mb/it}{}
            \DeclareFontShape{T1}{pzc}{mb}{sl}{<->ssub * pzc/mb/it}{}
            \DeclareFontShape{T1}{pzc}{m}{sl}{<->ssub * pzc/mb/sl}{}
            \DeclareFontShape{T1}{pzc}{m}{n}{<->ssub * pzc/mb/it}{}
            \usepackage{chancery}
            \usepackage{mathastext}
            \linespread{1.05}
            \begin{document}\boldmath
            """,
        },
    "gfsbodoni": {
        "description": "GFS Bodoni",
        "config": r"""
            \usepackage[T1]{fontenc}
            \renewcommand{\rmdefault}{bodoni}
            \usepackage[LGRgreek]{mathastext}
            \let\varphi\phi
            \linespread{1.06}
            """,
        },
    "palatinosg": {
        "description": "Palatino (Symbol Greek)",
        "config": r"""
            \usepackage[T1]{fontenc}
            \usepackage{palatino}
            \usepackage[symbolmax,defaultmathsizes]{mathastext}
            """,
        },
    "ncssgpxm": {
        "description": "New Century Schoolbook (Symbol Greek, PX math symbols)",
        "config": r"""
            \usepackage[T1]{fontenc}
            \usepackage{pxfonts}
            \usepackage{newcent}
            \usepackage[symbolgreek,defaultmathsizes]{mathastext}
            \linespread{1.06}
            """,
        },
    "epigrafica": {
        "description": "Epigrafica",
        "config": r"""
            \usepackage[LGR,OT1]{fontenc}
            \usepackage{epigrafica}
            \usepackage[basic,LGRgreek,defaultmathsizes]{mathastext}
            \let\varphi\phi
            \linespread{1.2}
            """,
        },
    "gfsneohellenic": {
        "description": "GFS NeoHellenic",
        "config": r"""
            \usepackage[T1]{fontenc}
            \renewcommand{\rmdefault}{neohellenic}
            \usepackage[LGRgreek]{mathastext}
            \let\varphi\phi
            \linespread{1.06}
            """,
        },
    "comfortaa": {
        "description": "Comfortaa",
        "config": r"""
            \usepackage[default]{comfortaa}
            \usepackage[LGRgreek,defaultmathsizes,noasterisk]{mathastext}
            \let\varphi\phi
            \linespread{1.06}
            """,
        },
    "slitexeg": {
        "description": "SliTeX (Euler Greek)",
        "config": r"""
            \usepackage[T1]{fontenc}
            \usepackage{tpslifonts}
            \usepackage[eulergreek,defaultmathsizes]{mathastext}
            \MTEulerScale{1.06}
            \linespread{1.2}
            """,
        },
    # "aptxgm": {
    #     "description": "Antykwa Półtawskiego (TX Fonts for Greek and math symbols)",
    #     "config": r"""
    #         \usepackage[OT4,OT1]{fontenc}
    #         \usepackage{txfonts}
    #         \usepackage[upright]{txgreeks}
    #         \usepackage{antpolt}
    #         \usepackage[defaultmathsizes,nolessnomore]{mathastext}
    #         """,
    #     },
    "baskervaldadff": {
        "description": "Baskervald ADF with Fourier",
        "config": r"""
            \usepackage[upright]{fourier}
            \usepackage{baskervald}
            \usepackage[defaultmathsizes,noasterisk]{mathastext}
            """,
        },
    "libertine": {
        "description": "Libertine",
        "config": r"""
            \usepackage[T1]{fontenc}
            \usepackage{libertine}
            \usepackage[greek=n]{libgreek}
            \usepackage[noasterisk,defaultmathsizes]{mathastext}
            """,
        },
    "biolinum": {
        "description": "Biolinum",
        "config": r"""
            \usepackage[T1]{fontenc}
            \usepackage{libertine}
            \renewcommand{\familydefault}{\sfdefault}
            \usepackage[greek=n,biolinum]{libgreek}
            \usepackage[noasterisk,defaultmathsizes]{mathastext}
            """,
        },
    # "mpmptx": {
    #     "description": "Minion Pro and Myriad Pro (and TX fonts symbols)",
    #     "config": r"""
    #         \usepackage{txfonts}
    #         \usepackage[upright]{txgreeks}
    #         \usepackage[no-math]{fontspec}
    #         \setmainfont[Mapping=tex-text]{Minion Pro}
    #         \setsansfont[Mapping=tex-text,Scale=MatchUppercase]{Myriad Pro}
    #         \renewcommand\familydefault\sfdefault
    #         \usepackage[defaultmathsizes]{mathastext}
    #         \renewcommand\familydefault\rmdefault
    #         """,
    #     "compiler": "xelatex"
    #     },
    # "mptx": {
    #     "description": "Minion Pro (and TX fonts symbols)",
    #     "config": r"""
    #         \usepackage{txfonts}
    #         \usepackage[no-math]{fontspec}
    #         \setmainfont[Mapping=tex-text]{Minion Pro}
    #         \usepackage[defaultmathsizes]{mathastext}
    #         """,
    #     "compiler": "xelatex"
    #     },
    "gnufsfs": {
        "description": "GNU FreeSerif and FreeSans",
        "config": r"""
            \usepackage[no-math]{fontspec}
            \setmainfont[ExternalLocation,
                         Mapping=tex-text,
                         BoldFont=FreeSerifBold,
                         ItalicFont=FreeSerifItalic,
                         BoldItalicFont=FreeSerifBoldItalic]{FreeSerif}
            \setsansfont[ExternalLocation,
                         Mapping=tex-text,
                         BoldFont=FreeSansBold,
                         ItalicFont=FreeSansOblique,
                         BoldItalicFont=FreeSansBoldOblique,
                         Scale=MatchLowercase]{FreeSans}
            \renewcommand{\familydefault}{lmss}
            \usepackage[LGRgreek,defaultmathsizes,noasterisk]{mathastext}
            \renewcommand{\familydefault}{\sfdefault}
            \Mathastext
            \let\varphi\phi % no `var' phi in LGR encoding
            \renewcommand{\familydefault}{\rmdefault}
            """,
        "compiler": "xelatex",
        "output": ".pdf"
        },
    "gnufstx": {
        "description": "GNU FreeSerif (and TX fonts symbols)",
        "config": r"""
            \usepackage[no-math]{fontspec}
            \usepackage{txfonts}  %\let\mathbb=\varmathbb
            \setmainfont[ExternalLocation,
                         Mapping=tex-text,
                         BoldFont=FreeSerifBold,
                         ItalicFont=FreeSerifItalic,
                         BoldItalicFont=FreeSerifBoldItalic]{FreeSerif}
            \usepackage[defaultmathsizes]{mathastext}
            """,
        "compiler": "xelatex",
        "output": ".pdf"
        },
    "librisadff": {
        "description": "Libris ADF with Fourier",
        "config": r"""
            \usepackage[T1]{fontenc}
            \usepackage[upright]{fourier}
            \usepackage{libris}
            \renewcommand{\familydefault}{\sfdefault}
            \usepackage[noasterisk]{mathastext}
            """,
        },
    # "vollkorntx": {
    #     "description": "Vollkorn (TX fonts for Greek and math symbols)",
    #     "config": r"""
    #         \usepackage[T1]{fontenc}
    #         \usepackage{txfonts}
    #         \usepackage[upright]{txgreeks}
    #         \usepackage{vollkorn}
    #         \usepackage[defaultmathsizes]{mathastext}
    #         """,
    #     },
    # "brushscriptxpx": {
    #     "description": "BrushScriptX-Italic (PX math and Greek)",
    #     "config": r"""
    #         \usepackage[T1]{fontenc}
    #         \usepackage{pxfonts}
    #         %\usepackage{pbsi}
    #         \renewcommand{\rmdefault}{pbsi}
    #         \renewcommand{\mddefault}{xl}
    #         \renewcommand{\bfdefault}{xl}
    #         \usepackage[defaultmathsizes,noasterisk]{mathastext}
    #         \begin{document}\boldmath
    #         """,
    #         "compiler": "xelatex"
    #     },
    "ecftallpaul": {
        "description": "ECF Tall Paul (with Symbol font)",
        "config": r"""
            \DeclareFontFamily{T1}{ftp}{}
            \DeclareFontShape{T1}{ftp}{m}{n}{
               <->s*[1.4] ftpmw8t
            }{} % increase size by factor 1.4
            \renewcommand\familydefault{ftp} % emerald package
            \usepackage[symbol]{mathastext}
            \let\infty\inftypsy
            """,
        },
    "ecfaugieeg": {
        "description": "ECF Augie (Euler Greek)",
        "config": r"""
            \renewcommand\familydefault{fau} % emerald package
            \usepackage[defaultmathsizes,eulergreek]{mathastext}
            """,
        "compiler": "pdflatex",
        "output": ".pdf"
        },
    "ecfjdtx": {
        "description": "ECF JD (with TX fonts)",
        "config": r"""
            \usepackage{txfonts}
            \usepackage[upright]{txgreeks}
            \renewcommand\familydefault{fjd} % emerald package
            \usepackage{mathastext}
            \begin{document}\mathversion{bold}
            """,
        },
    "ecfwebstertx": {
        "description": "ECF Webster (with TX fonts)",
        "config": r"""
            \usepackage{txfonts}
            \usepackage[upright]{txgreeks}
            \renewcommand\familydefault{fwb} % emerald package
            \usepackage{mathastext}
            \renewcommand{\int}{\intop\limits}
            \linespread{1.5}
            \begin{document}\mathversion{bold}
            """,
        },
    # "comicsansms": {
    #     "description": "Comic Sans MS",
    #     "config": r"""
    #         \usepackage[no-math]{fontspec}
    #         \setmainfont[Mapping=tex-text]{Comic Sans MS}
    #         \usepackage[defaultmathsizes]{mathastext}
    #         """,
    #     "compiler": "xelatex",
    #     },
    "electrumadfcm": {
        "description": "Electrum ADF (CM Greek)",
        "config": r"""
            \usepackage[T1]{fontenc}
            \usepackage[LGRgreek,basic,defaultmathsizes]{mathastext}
            \usepackage[lf]{electrum}
            \Mathastext
            \let\varphi\phi
            """,
        },
    # "americantypewriter": {
    #     "description": "American Typewriter",
    #     "config": r"""
    #         \usepackage[no-math]{fontspec}
    #         \setmainfont[Mapping=tex-text]{American Typewriter}
    #         \usepackage[defaultmathsizes]{mathastext}
    #         """,
    #     "compiler": "xelatex"
    #     },
    # "papyrus": {
    #     "description": "Papyrus",
    #     "config": r"""
    #         \usepackage[no-math]{fontspec}
    #         \setmainfont[Mapping=tex-text]{Papyrus}
    #         \usepackage[defaultmathsizes]{mathastext}
    #         """,
    #     "compiler": "xelatex"
    #     },
    # "noteworthylight": {
    #     "description": "Noteworthy Light",
    #     "config": r"""
    #         \usepackage[no-math]{fontspec}
    #         \setmainfont[Mapping=tex-text]{Noteworthy Light}
    #         \usepackage[defaultmathsizes]{mathastext}
    #         """,
    #     "compiler": "xelatex"
    #     },
    # "chalkboardse": {
    #     "description": "Chalkboard SE",
    #     "config": r"""
    #         \usepackage[no-math]{fontspec}
    #         \setmainfont[Mapping=tex-text]{Chalkboard SE}
    #         \usepackage[defaultmathsizes]{mathastext}
    #         """,
    #     "compiler": "xelatex"
    #     },
    "chalkduster": {
        "description": "Chalkduster",
        "config": r"""
            \usepackage[no-math]{fontspec}
            \setmainfont[Mapping=tex-text]{Chalkduster}
            \usepackage[defaultmathsizes]{mathastext}
            """,
        "compiler": "xelatex",
        "output": ".xdv"
        },
    # # "applechancery": {
    # #     "description": "Apple Chancery",
    # #     "config": r"""
    # #         \usepackage[no-math]{fontspec}
    # #         \setmainfont[Mapping=tex-text]{Apple Chancery}
    # #         \usepackage[defaultmathsizes]{mathastext}
    # #         """,
    # #     "compiler": "xelatex"
    # #     },
    # "zapfchancery": {
    #     "description": "Zapf Chancery",
    #     "config": r"""
    #         \DeclareFontFamily{T1}{pzc}{}
    #         \DeclareFontShape{T1}{pzc}{mb}{it}{<->s*[1.2] pzcmi8t}{}
    #         \DeclareFontShape{T1}{pzc}{m}{it}{<->ssub * pzc/mb/it}{}
    #         \usepackage{chancery} % = \renewcommand{\rmdefault}{pzc}
    #         \renewcommand\shapedefault\itdefault
    #         \renewcommand\bfdefault\mddefault
    #         \usepackage[defaultmathsizes]{mathastext}
    #         \linespread{1.05}
    #         """,
    #     },
    # # "italicvollkornf": {
    # #     "description": "Vollkorn with Fourier (Italic)",
    # #     "config": r"""
    # #         \usepackage{fourier}
    # #         \usepackage{vollkorn}
    # #         \usepackage[italic,nohbar]{mathastext}
    # #         """,
    # #     },
    # "italiclmtpcm": {
    #     "description": "Latin Modern Typewriter Proportional (CM Greek) (Italic)",
    #     "config": r"""
    #         \usepackage[T1]{fontenc}
    #         \usepackage[variablett,nomath]{lmodern}
    #         \renewcommand{\familydefault}{\ttdefault}
    #         \usepackage[frenchmath]{mathastext}
    #         \linespread{1.08}
    #         """,
    #     },
    # "italictimesf": {
    #     "description": "Times with Fourier (Italic)",
    #     "config": r"""
    #         \usepackage{fourier}
    #         \renewcommand{\rmdefault}{ptm}
    #         \usepackage[italic,defaultmathsizes,noasterisk]{mathastext}
    #         """,
    #     },
    # "italichelveticaf": {
    #     "description": "Helvetica with Fourier (Italic)",
    #     "config": r"""
    #         \usepackage[T1]{fontenc}
    #         \usepackage[scaled]{helvet}
    #         \usepackage{fourier}
    #         \renewcommand{\rmdefault}{phv}
    #         \usepackage[italic,defaultmathsizes,noasterisk]{mathastext}
    #         """,
    #     },
    # "italicvanturisadff": {
    #     "description": "Venturis ADF with Fourier (Italic)",
    #     "config": r"""
    #         \usepackage{fourier}
    #         \usepackage[lf]{venturis}
    #         \usepackage[italic,defaultmathsizes,noasterisk]{mathastext}
    #         """,
    #     },
    # "italicromandeadff": {
    #     "description": "Romande ADF with Fourier (Italic)",
    #     "config": r"""
    #         \usepackage[T1]{fontenc}
    #         \usepackage{fourier}
    #         \usepackage{romande}
    #         \usepackage[italic,defaultmathsizes,noasterisk]{mathastext}
    #         \renewcommand{\itshape}{\swashstyle}
    #         """,
    #     },
    # "italicgfsdidot": {
    #     "description": "GFS Didot (Italic)",
    #     "config": r"""
    #         \usepackage[T1]{fontenc}
    #         \renewcommand\rmdefault{udidot}
    #         \usepackage[LGRgreek,defaultmathsizes,italic]{mathastext}
    #         \let\varphi\phi
    #         """,
    #     },
    # "italicdroidserifpx": {
    #     "description": "Droid Serif (PX math symbols) (Italic)",
    #     "config": r"""
    #         \usepackage[T1]{fontenc}
    #         \usepackage{pxfonts}
    #         \usepackage[default]{droidserif}
    #         \usepackage[LGRgreek,defaultmathsizes,italic,basic]{mathastext}
    #         \let\varphi\phi
    #         """,
    #     },
    # "italicdroidsans": {
    #     "description": "Droid Sans (Italic)",
    #     "config": r"""
    #         \usepackage[T1]{fontenc}
    #         \usepackage[default]{droidsans}
    #         \usepackage[LGRgreek,defaultmathsizes,italic]{mathastext}
    #         \let\varphi\phi
    #         """,
    #     },
    # "italicverdana": {
    #     "description": "Verdana (Italic)",
    #     "config": r"""
    #         \usepackage[no-math]{fontspec}
    #         \setmainfont[Mapping=tex-text]{Verdana}
    #         \usepackage[defaultmathsizes,italic]{mathastext}
    #         """,
    #     "compiler": "xelatex"
    #     },
    # "italicbaskerville": {
    #     "description": "Baskerville (Italic)",
    #     "config": r"""
    #         \usepackage[no-math]{fontspec}
    #         \setmainfont[Mapping=tex-text]{Baskerville}
    #         \usepackage[defaultmathsizes,italic]{mathastext}
    #         """,
    #     "compiler": "xelatex"
    #     },
}