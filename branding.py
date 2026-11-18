"""Institutional logo display with consistent spacing and responsive sizing."""
from pathlib import Path
import tkinter as tk

from PIL import Image, ImageTk


class ResponsiveLogo(tk.Canvas):
    def __init__(self, parent, background, width, padding=(16, 12), rounded=True):
        super().__init__(parent, bg=background, bd=0, highlightthickness=0)
        with Image.open(Path(__file__).parent/'assets'/'kmitl-siet.png') as source:
            self.source = source.copy()
        self.padding = padding
        self.rounded = rounded
        self.logo_width = None
        self.set_width(width)

    def set_width(self, width):
        width = max(1, round(width))
        if width == self.logo_width:return
        self.logo_width = width
        height = max(1, round(width*self.source.height/self.source.width))
        px, py = self.padding
        w, h = width+px*2, height+py*2
        self.configure(width=w, height=h)
        self.delete('all')
        if self.rounded:
            radius = 10
            self.create_rectangle(radius, 0, w-radius, h, fill='white', outline='')
            self.create_rectangle(0, radius, w, h-radius, fill='white', outline='')
            for x, y in ((0, 0), (w-radius*2, 0), (0, h-radius*2), (w-radius*2, h-radius*2)):
                self.create_oval(x, y, x+radius*2, y+radius*2, fill='white', outline='')
        self.photo = ImageTk.PhotoImage(self.source.resize((width, height), Image.Resampling.LANCZOS), master=self)
        self.create_image(px, py, anchor='nw', image=self.photo)

    def follow_width(self, container, fraction, minimum, maximum, inset=0):
        def resize(event):
            available = max(1, event.width-inset)
            self.set_width(min(maximum, max(minimum, available*fraction)))
        container.bind('<Configure>', resize, add='+')
