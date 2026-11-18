"""Board overview and checklist, with a manual selection option."""
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog

from PIL import Image, ImageTk
from board import COMPONENTS
from ui import COLORS

STATUS_TEXT={'PASS':'ตรวจครบ · ปกติ','FAIL':'พบข้อบกพร่อง','REVIEW':'รอตรวจเพิ่ม','PENDING':'ยังไม่ตรวจ'}


class BoardWindow:
    def __init__(self,app):
        self.app=app
        self.window=tk.Toplevel(app.root)
        self.window.title('แผนที่บอร์ดและเช็กลิสต์อุปกรณ์')
        self.window.geometry('1000x740')
        self.window.minsize(920,680)
        self.window.configure(bg=COLORS['bg'])
        tk.Label(self.window,text='แผนที่บอร์ดและเช็กลิสต์ผลตรวจ',font=(app.ui_font,16,'bold'),
                 fg=COLORS['ink'],bg=COLORS['bg']).pack(anchor='w',padx=20,pady=(18,4))
        self.summary=tk.StringVar()
        tk.Label(self.window,textvariable=self.summary,bg=COLORS['bg'],fg=COLORS['accent'],font=(app.ui_font,10)).pack(anchor='w',padx=20)
        body=tk.Frame(self.window,bg=COLORS['bg'])
        body.pack(fill='both',expand=True,padx=20,pady=12)
        left=tk.Frame(body,bg='white');left.pack(side='left',fill='y',padx=(0,16))
        self.map=tk.Canvas(left,width=431,height=447,bg='white',highlightthickness=0)
        self.map.pack(padx=6,pady=6)
        self.photo=ImageTk.PhotoImage(Image.open(Path(__file__).parent/'assets'/'smd-board-reference.png'))
        self.map.bind('<Button-1>',self.click_map)
        tk.Label(left,text='สีเขียว: ผ่าน · สีแดง: ไม่ผ่าน\nสีส้ม: ตรวจไม่ครบ · สีเทา: รอตรวจ\nคลิกอุปกรณ์บนผังเพื่อดูหลักฐาน\n\nโหมดอัตโนมัติจับตำแหน่งจากภาพอ้างอิง\nโหมดตรวจเองเลือกชื่ออุปกรณ์จากแผนที่',
                 bg='white',fg=COLORS['muted'],justify='left',font=(app.ui_font,10)).pack(anchor='w',padx=15,pady=12)
        right=tk.Frame(body,bg='white');right.pack(side='left',fill='both',expand=True)
        self.table=ttk.Treeview(right,columns=('name','state','coverage'),show='headings',selectmode='browse')
        for col,title,width in [('name','อุปกรณ์',75),('state','สถานะ',160),('coverage','จุดตรวจ',75)]:
            self.table.heading(col,text=title);self.table.column(col,width=width,anchor='center')
        self.table.pack(side='left',fill='both',expand=True)
        scroll=ttk.Scrollbar(right,command=self.table.yview);scroll.pack(side='right',fill='y')
        self.table.configure(yscrollcommand=scroll.set)
        for name,color in [('PASS',COLORS['good']),('FAIL',COLORS['red']),('REVIEW',COLORS['amber'])]:
            self.table.tag_configure(name,foreground=color)
        self.table.bind('<<TreeviewSelect>>',self.click_row)
        actions=tk.Frame(self.window,bg=COLORS['bg']);actions.pack(fill='x',padx=20,pady=(0,16))
        for text,command in [('บอร์ดใหม่',self.new_board),('ส่งออกเช็กลิสต์ CSV',self.export),('เลือกตัวถัดไป',self.next_component),('กลับไปตรวจภาพ',self.window.destroy)]:
            tk.Button(actions,text=text,command=command,bg=COLORS['accent'],fg='white',relief='flat',padx=14,pady=10).pack(side='left',padx=(0,8))
        self.refresh()

    def refresh(self):
        if not self.window.winfo_exists():return
        ledger=self.app.checklist
        self.summary.set(f"บอร์ด {ledger.session['board_id']}  ·  ตรวจครบ {ledger.completed()}/{len(COMPONENTS)} อุปกรณ์")
        self.table.delete(*self.table.get_children())
        self.map.delete('all');self.map.create_image(0,0,image=self.photo,anchor='nw')
        for name,data in COMPONENTS.items():
            item=ledger.session['results'].get(name,{})
            state=item.get('status','PENDING')
            self.table.insert('','end',iid=name,values=(name,STATUS_TEXT[state],f"{item.get('covered',0)}/{data['expected']}"),tags=(state,))
            color={'PASS':COLORS['good'],'FAIL':COLORS['red'],'REVIEW':COLORS['amber']}.get(state,'#91a0ad')
            selected=name==self.app.component.get()
            self.map.create_rectangle(*data['box'],outline='#ffffff' if selected else color,width=3 if selected else 1)
        current=self.app.component.get()
        if current in COMPONENTS:
            self.table.selection_set(current);self.table.see(current)

    def click_map(self,event):
        for name,data in COMPONENTS.items():
            x1,y1,x2,y2=data['box']
            if x1<=event.x<=x2 and y1<=event.y<=y2:
                self.select(name)
                self.app.show_evidence(name)
                self.refresh()
                break

    def click_row(self,event):
        selection=self.table.selection()
        if selection and selection[0]!=self.app.component.get():
            self.select(selection[0]);self.refresh()

    def select(self,name):
        if self.app.auto_mode.get():
            self.app.component.set(name)
            self.app.refresh_checklist()
        else:self.app.select_component(name)

    def next_component(self):
        for name in COMPONENTS:
            if not self.app.checklist.session['results'].get(name,{}).get('complete',False):
                self.select(name);self.refresh();return

    def new_board(self):
        self.app.begin_new_board()

    def export(self):
        path=filedialog.asksaveasfilename(parent=self.window,defaultextension='.csv',initialfile='checklist_'+self.app.checklist.session['board_id']+'.csv',filetypes=[('CSV','*.csv')])
        if path:
            try:
                self.app.checklist.export_csv(path)
                self.app.info.set('ส่งออกเช็กลิสต์แล้ว: '+path)
            except Exception as exc:self.app.error(exc)
