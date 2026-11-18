"""Shared Thai UI typography, using fonts already installed on the computer."""
from tkinter import font
from tkinter import TclError

LATIN_FONT = 'Segoe UI'


def fit_single_line(label, family, max_size, bold=False):
    """Keep the complete label on one line, using the largest font that fits."""
    text = ' '.join(label.cget('text').splitlines())
    fitted = font.Font(root=label, family=family, size=max_size,
                       weight='bold' if bold else 'normal')
    label.configure(text=text, font=fitted, wraplength=0, padx=0, borderwidth=0,
                    highlightthickness=0)
    label.fitted_font = fitted

    def resize(event):
        available = event.width - 4
        if available <= 0:return
        low, high = 1, max_size
        while low < high:
            candidate = (low + high + 1) // 2
            fitted.configure(size=candidate)
            if fitted.measure(text) <= available:low = candidate
            else:high = candidate - 1
        if fitted.actual('size') != low:fitted.configure(size=low)

    label.bind('<Configure>', resize, add='+')
    return fitted


def configure_typography(root):
    available = set(font.families(root))
    family = next((name for name in ('Leelawadee UI', 'Leelawadee', 'Tahoma')
                   if name in available), font.nametofont('TkDefaultFont', root=root).actual('family'))
    for name in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont', 'TkHeadingFont',
                 'TkCaptionFont', 'TkSmallCaptionFont', 'TkIconFont', 'TkTooltipFont'):
        try:
            font.nametofont(name, root=root).configure(family=family)
        except TclError:
            # Some Tk platforms don't define every named UI font.
            pass
    root.option_add('*Font', (family, 10))
    return family
