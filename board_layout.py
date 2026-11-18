"""Interactive board diagram backed by actual component inspection records."""
import tkinter as tk
from PIL import Image, ImageOps, ImageTk

from board import COMPONENTS

# Diagram positions follow the physical PCB. Calibration uses separate geometry.
BOARD_BOUNDS=(122,109,880,917)
DISPLAY_BOXES={}
for row,y in enumerate((261,331,400,470,539,609,678,748,817,887)):
    for name,left,right in ((f'R{row+11}',135,216),(f'R{row+1}',224,308),
                            (f'R{40-row}',325,385),(f'R{row+21}',393,453)):
        DISPLAY_BOXES[name]=(left,y-42,right,y+12)
for row,(top,bottom) in enumerate(((211,289),(292,365),(367,440),(442,514),(516,589),
                                  (592,668),(671,744),(747,821),(824,903))):
    DISPLAY_BOXES[f'R{row+41}']=(472,top+8,573,bottom-8)
    if row<5:DISPLAY_BOXES[f'D{row+1}']=(594,top+8,700,bottom-8)
for name,box in zip(('Q1','Q2','Q3','Q4','Q5','Q6'),
                    ((732,502,787,590),(798,502,856,590),(590,610,647,704),
                     (658,610,717,704),(729,610,787,704),(798,610,856,704))):
    DISPLAY_BOXES[name]=box
DISPLAY_BOXES.update(U1=(729,211,858,446),U4=(595,727,752,904))

class BoardDiagram(tk.Canvas):
    PALETTE = {'PENDING':('#23394a','#7e91a0'), 'SEEN':('#134a66','#76d2ff'),
               'PASS':('#0d5c51','#72e1ba'), 'FAIL':('#762f40','#ffb6c1'),
               'REVIEW':('#6a4b20','#ffda8a')}

    def __init__(self, parent, on_select):
        super().__init__(parent,bg='white',highlightthickness=0,width=340,height=355)
        self.on_select=on_select
        self.hit_boxes={}
        self.states={}
        self.selected=None
        self.tiles={}
        self.photos=[]
        self.board_number=''
        self.bind('<Configure>',lambda event:self.draw())
        self.bind('<Button-1>',self.select)
        self.bind('<Double-1>',lambda event:'break')
        self.bind('<Motion>',self.hover)
        self.bind('<MouseWheel>',lambda event:self.yview_scroll(-int(event.delta/120),'units'))

    def draw(self):
        self.delete('all');self.hit_boxes={}
        self.photos.clear()
        if not self.states:return
        width,height=self.winfo_width(),self.winfo_height()
        bw,bh=BOARD_BOUNDS[2]-BOARD_BOUNDS[0],BOARD_BOUNDS[3]-BOARD_BOUNDS[1]
        scale=min(max(1,width-12)/bw,max(290,height-12)/bh)
        ox,oy=(width-bw*scale)/2,max(6,(height-bh*scale)/2)
        self.configure(scrollregion=(0,0,width,max(height,bh*scale+12)))
        def box(coords):
            x1,y1,x2,y2=coords
            return (ox+(x1-BOARD_BOUNDS[0])*scale,oy+(y1-BOARD_BOUNDS[1])*scale,
                    ox+(x2-BOARD_BOUNDS[0])*scale,oy+(y2-BOARD_BOUNDS[1])*scale)
        def text(x,y,value,size=20,color='#b8d1df',anchor='center'):
            cx,cy,_,_=box((x,y,x,y))
            return self.create_text(cx,cy,text=value,fill=color,anchor=anchor,
                                    font=('Segoe UI',-max(8,round(size*scale)),'bold'))
        self.create_rectangle(*box(BOARD_BOUNDS),fill='#102c40',outline='#648698',width=2)
        text(144,137,'SMD · INSPECTION BOARD',20,'#e0eef4','w')
        text(618,137,'No.',20,'#e0eef4','e')
        self.create_rectangle(*box((630,113,872,161)),fill='white',outline='#b8d1df',width=1)
        if self.board_number:
            number=self.board_number
            if len(number)>18:number=number[:8]+'…'+number[-8:]
            number_item=text(751,137,number,32,'#102c40')
            bounds=self.bbox(number_item)
            if bounds[2]-bounds[0]>220*scale:
                size=max(6,round(32*scale*220*scale/(bounds[2]-bounds[0])))
                self.itemconfigure(number_item,font=('Segoe UI',-size,'bold'))
        for region in ((126,164,315,911),(315,164,460,911),(714,164,875,452),
                       (714,452,875,600),(585,600,875,712),(585,712,875,911)):
            self.create_rectangle(*box(region),outline='#35556a')
        outline=[]
        for x,y in ((460,164),(714,164),(714,600),(585,600),(585,911),(460,911)):
            cx,cy,_,_=box((x,y,x,y));outline.extend((cx,cy))
        self.create_polygon(*outline,fill='',outline='#35556a')
        for x,y,value in ((220,188,'0805'),(388,188,'0603'),(584,188,'1206'),
                          (797,188,'SOP14'),(797,478,'SOT-23'),(813,750,'SSOP28')):
            text(x,y,value)
        for name in COMPONENTS:
            coords=box(DISPLAY_BOXES[name]);self.hit_boxes[name]=coords
            fill,stroke=self.PALETTE.get(self.states[name],self.PALETTE['PENDING'])
            self.create_rectangle(*coords,fill=fill,outline=stroke,width=2)
            if name in self.tiles and min(coords[2]-coords[0],coords[3]-coords[1])>=12:
                a,b,c,d=coords
                area=(max(1,round(c-a)-4),max(1,round(d-b)-4))
                photo=ImageTk.PhotoImage(ImageOps.contain(self.tiles[name],area,Image.Resampling.BILINEAR),master=self)
                self.photos.append(photo)
                self.create_rectangle(a+1,b+1,c-1,d-1,fill=fill,outline='')
                self.create_image((a+c)/2,(b+d)/2,image=photo,tags=('thumbnail',name))
            else:
                x1,y1,x2,y2=DISPLAY_BOXES[name]
                text((x1+x2)/2,(y1+y2)/2,name,22,stroke)
            if name==self.selected:
                a,b,c,d=coords
                self.create_rectangle(a-3,b-3,c+3,d+3,outline='white',dash=(3,2))

    def component_at(self,event):
        x,y=self.canvasx(event.x),self.canvasy(event.y)
        for name,(x1,y1,x2,y2) in self.hit_boxes.items():
            if x1<=x<=x2 and y1<=y<=y2:return name

    def select(self,event):
        name=self.component_at(event)
        if name:self.on_select(name)

    def hover(self,event):
        self.configure(cursor='hand2' if self.component_at(event) else '')


class BoardLayout(BoardDiagram):
    def __init__(self,parent,app):
        super().__init__(parent,self.open_component)
        self.app=app

    def refresh(self, force=False):
        records=self.app.checklist.session['results']
        seen={t.component for t in self.app.auto_targets}
        seen.update(hit.component for hit in self.app.position_hits if hit.component in COMPONENTS)
        states={name:records.get(name,{}).get('status','SEEN' if name in seen else 'PENDING') for name in COMPONENTS}
        selected=self.app.component.get()
        number=self.app.board_number.get().strip()
        if not force and states==self.states and selected==self.selected and number==self.board_number:return
        self.board_number=number
        self.states,self.selected=states,selected
        self.draw()

    def open_component(self,name):
        if self.app.auto_mode.get():
            self.app.select_component_info(name)
        elif not self.app.busy:
            self.app.select_component(name);self.app.select_component_info(name)

