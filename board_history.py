"""Read completed-board exports without switching the board being inspected."""
import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk,messagebox
from send2trash import send2trash

import cv2
from PIL import Image, ImageOps

from board import COMPONENTS
from board_layout import BoardDiagram
from evidence import crop_evidence
from ic_evidence import bank_photo, bank_reports
from inspection import read_image
from component_info import failure_reason

STATUS_COLOURS={'PASS':'#087f73','FAIL':'#d14d57','REVIEW':'#b67b21'}


def recycle_board(folder,key):
    root=Path(folder).resolve()
    target=root/key
    resolved=target.resolve()
    if (not key.startswith('board_') or Path(key).name!=key or
        resolved.parent!=root or resolved.name!=key or target.is_symlink() or
        not (target/'board.json').is_file()):
        raise ValueError('ตำแหน่งโฟลเดอร์บอร์ดไม่ถูกต้อง')
    send2trash(str(target))


def archive_entry(folder, entry):
    def resolve(item):
        result = dict(item)
        if result.get('report'):
            result['report'] = str(folder / result['report'])
        return result
    result = resolve(entry)
    result['coverage'] = {pin:resolve(item) for pin,item in entry.get('coverage',{}).items()}
    return result


def component_tile(name, entry):
    """Use photos from the selected snapshot, before inspection overlays when available."""
    if name in ('U1','U4'):
        banks = bank_reports(name,entry)
        vertical = name=='U1'
        tile = Image.new('RGB',(220,340) if vertical else (300,260),'#102c40')
        found = False
        for side,reports in enumerate(banks):
            for folder,pins in reports:
                try:
                    frame = bank_photo(folder,name,pins,raw=True)
                except (OSError,ValueError,KeyError,TypeError,cv2.error):
                    continue
                view = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                if (view.height>view.width) != vertical:
                    view=view.transpose(Image.Transpose.ROTATE_90)
                area=(90,340) if vertical else (300,110)
                view=ImageOps.contain(view,area)
                origin=(side*130,0) if vertical else (0,side*150)
                tile.paste(view,(origin[0]+(area[0]-view.width)//2,origin[1]+(area[1]-view.height)//2))
                found = True
                break
        if found:return tile
    if not entry.get('report'):return None
    folder = Path(entry['report'])
    meta = json.loads((folder/'result.json').read_text('utf-8'))
    if meta.get('component',name) != name:return None
    if (folder/'raw.png').is_file() and (meta.get('roi_quad') or meta.get('roi')):
        frame = read_image(folder/'raw.png')
        if meta.get('roi_quad'):
            frame = crop_evidence(frame,meta['roi_quad'],padding=0)
        else:
            x1,y1,x2,y2 = map(int,meta['roi'])
            frame = frame[max(0,y1):max(0,y2),max(0,x1):max(0,x2)]
    else:
        path = next((folder/f for f in ('crop.png','component_result.png') if (folder/f).is_file()),None)
        if path is None:return None
        frame = read_image(path)
    tile = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    tile.thumbnail((384,384),Image.Resampling.LANCZOS)
    return tile


class BoardHistory(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=14)
        self.app = app
        self.folder = app.report_root / 'completed_boards'
        self.entries = {}
        self.messages = queue.Queue()
        self.loading = False
        self.deleting = False
        self.delete_messages = queue.Queue()
        self.tile_worker = None
        self.timers=set()
        self.dirty = True
        self.note = tk.StringVar()
        header = ttk.Frame(self)
        header.pack(fill='x', pady=(0, 8))
        ttk.Label(header, textvariable=self.note).pack(side='left')
        self.reload_button=ttk.Button(header,text='โหลดประวัติใหม่',command=self.reload)
        self.reload_button.pack(side='right')
        self.style=ttk.Style(self)
        self.style.configure('Delete.TButton',foreground='#d14d57',padding=(12,8))
        self.style.map('Delete.TButton',foreground=[('disabled','#a4b4bd')])
        self.delete_button=ttk.Button(header,text='ลบบอร์ดที่เลือก',command=self.delete_selected,
                                     style='Delete.TButton',state='disabled')
        self.delete_button.pack(side='right',padx=(0,8))
        columns = ttk.Panedwindow(self,orient='horizontal')
        columns.pack(fill='both',expand=True)
        self.boards = self.table(columns,'บอร์ดที่บันทึกแล้ว',
                                 [('number','เลขบอร์ด',130),('time','เวลาบันทึก',170),('status','ผล',70)])
        self.boards.configure(style='BoardHistory.Treeview')
        for state,colour in STATUS_COLOURS.items():self.boards.tag_configure(state,foreground=colour)
        panel = ttk.Frame(columns,padding=6)
        columns.add(panel,weight=3)
        self.caption = tk.StringVar(value='เลือกบอร์ดเพื่อดูภาพอุปกรณ์')
        ttk.Label(panel,textvariable=self.caption).pack(anchor='w',pady=5)
        body=ttk.Frame(panel)
        body.pack(fill='both',expand=True)
        side=ttk.Frame(body,padding=(12,4))
        side.pack(side='right',fill='y')
        self.failure_title=tk.StringVar(value='ผลตรวจที่ต้องตรวจสอบ')
        ttk.Label(side,textvariable=self.failure_title).pack(anchor='w',pady=(0,8))
        self.failure_details=tk.Text(side,width=27,wrap='word',bd=0,bg='white',fg='#192c3e',
                                     font=(getattr(app,'ui_font','Segoe UI'),10),state='disabled')
        failure_scroll=ttk.Scrollbar(side,command=self.failure_details.yview)
        failure_scroll.pack(side='right',fill='y')
        self.failure_details.configure(yscrollcommand=failure_scroll.set)
        self.failure_details.pack(fill='both',expand=True)
        self.failure_details.tag_configure('FAIL',foreground='#d14d57')
        self.failure_details.tag_configure('REVIEW',foreground='#b67b21')
        self.diagram = BoardDiagram(body,self.open_part)
        self.diagram.pack(fill='both',expand=True)
        self.current_key = None
        self.records = {}
        self.cancel_tiles = threading.Event()
        self.bind('<Destroy>',self.on_destroy)
        self.boards.bind('<<TreeviewSelect>>',self.select_board)

    def update_actions(self):
        available=bool(self.boards.selection()) and not self.loading and not self.deleting
        self.delete_button.configure(state='normal' if available else 'disabled')
        self.reload_button.configure(state='disabled' if self.loading or self.deleting else 'normal')

    def delete_selected(self):
        selected=self.boards.selection()
        if self.loading or self.deleting or not selected:return
        key=selected[0]
        if key not in self.entries:return
        number=self.entries[key]['board_number']
        if not messagebox.askyesno('ลบบอร์ดที่เลือก',
                f'ย้ายบอร์ด {number} พร้อมภาพและรายงานทั้งหมดไปถังขยะหรือไม่?',
                parent=self.winfo_toplevel(),icon='warning'):return
        self.deleting=True
        self.cancel_tiles.set()
        worker=self.tile_worker
        self.update_actions()
        self.note.set('กำลังลบบอร์ด '+str(number)+'…')
        def remove():
            error=None
            try:
                if worker is not None:worker.join()
                recycle_board(self.folder,key)
            except Exception as exc:error=str(exc)
            self.delete_messages.put(error)
        def complete():
            try:error=self.delete_messages.get_nowait()
            except queue.Empty:
                self.schedule(40,complete)
                return
            self.deleting=False
            if error:
                messagebox.showerror('ลบบอร์ดไม่สำเร็จ',error,parent=self.winfo_toplevel())
            else:
                self.app.info.set('ย้ายบอร์ด '+str(number)+' ไปถังขยะแล้ว')
            self.reload()
        threading.Thread(target=remove,daemon=True,name='delete-board-history').start()
        self.schedule(40,complete)

    def show_failures(self,reasons):
        failed=sum(entry.get('status')=='FAIL' for entry in self.records.values())
        self.failure_title.set(f'ไม่ผ่าน {failed} อุปกรณ์')
        self.failure_details.configure(state='normal')
        self.failure_details.delete('1.0','end')
        for name,reason in reasons.items():
            state=self.records[name].get('status')
            title='MISSING' if self.records[name].get('component_missing') else ('ไม่ผ่าน' if state=='FAIL' else 'รอตรวจเพิ่ม')
            self.failure_details.insert('end',f'{name} · {title}\n',state)
            self.failure_details.insert('end',reason+'\n\n')
        if not reasons:self.failure_details.insert('end','ไม่พบรายการที่ไม่ผ่านหรือรอตรวจเพิ่ม' if self.records else 'เลือกบอร์ดเพื่อดูรายละเอียด')
        self.failure_details.configure(state='disabled')

    def on_destroy(self,event):
        if event.widget is self:
            self.cancel_tiles.set()
            for timer in self.timers:self.after_cancel(timer)
            self.timers.clear()

    def schedule(self,delay,callback):
        def run():
            self.timers.discard(timer)
            callback()
        timer=self.after(delay,run)
        self.timers.add(timer)

    @staticmethod
    def table(parent,title,columns):
        frame = ttk.Frame(parent,padding=6)
        parent.add(frame,weight=1)
        ttk.Label(frame,text=title).pack(anchor='w',pady=5)
        tree = ttk.Treeview(frame, columns=[col[0] for col in columns],show='headings',selectmode='browse')
        for key,label,width in columns:
            tree.heading(key,text=label)
            tree.column(key,width=width,anchor='center')
        scroll = ttk.Scrollbar(frame,command=tree.yview)
        scroll.pack(side='right',fill='y')
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(fill='both',expand=True)
        return tree

    def invalidate(self):
        self.dirty = True
        if self.winfo_ismapped():self.reload()

    def reload(self):
        if self.loading or self.deleting:return
        self.loading = True
        self.update_actions()
        self.dirty = False
        self.note.set('กำลังอ่านประวัติบอร์ด…')
        def read():
            entries = {}
            skipped = 0
            try:
                for path in sorted(self.folder.glob('board_*/board.json'),reverse=True):
                    try:
                        data = json.loads(path.read_text('utf-8'))
                        if not isinstance(data['results'],dict) or not data['board_number']:
                            raise ValueError('invalid board')
                        entries[path.parent.name] = {key:data[key] for key in ('board_number','saved_at','overall_status','results')}
                    except (OSError,ValueError,KeyError,TypeError):
                        skipped += 1
            except OSError:
                skipped += 1
            self.messages.put((entries,skipped))
        threading.Thread(target=read,daemon=True,name='completed-board-history').start()
        self.schedule(30,self.receive)

    def receive(self):
        try:self.entries,skipped = self.messages.get_nowait()
        except queue.Empty:
            self.schedule(30,self.receive)
            return
        self.loading = False
        previous = self.boards.selection()
        self.boards.delete(*self.boards.get_children())
        self.current_key = None
        for key,board in self.entries.items():
            self.boards.insert('','end',iid=key,values=(board['board_number'],str(board['saved_at'])[:19].replace('T',' '),board['overall_status']),
                               tags=(board['overall_status'],))
        if self.entries:
            self.boards.selection_set(previous[0] if previous and previous[0] in self.entries else next(iter(self.entries)))
        self.select_board()
        message = f'{len(self.entries)} บอร์ด' if self.entries else 'ยังไม่มีบอร์ดที่บันทึกเสร็จแล้ว'
        self.note.set(message + (f' · อ่านรายงานไม่ได้ {skipped} รายการ' if skipped else ''))
        if self.dirty:self.reload()

    def select_board(self,event=None):
        selected = self.boards.selection()
        key = selected[0] if selected else None
        self.update_actions()
        state=self.entries.get(key,{}).get('overall_status')
        colour=STATUS_COLOURS.get(state,'#192c3e')
        self.style.map('BoardHistory.Treeview',foreground=[('selected',colour)],
                       background=[('selected','#fff0f2' if state=='FAIL' else '#d8eee9')])
        if key and key==self.current_key:return
        self.cancel_tiles.set()
        self.current_key = key
        self.diagram.tiles = {}
        self.diagram.selected = None
        if key not in self.entries:
            self.records = {}
            self.show_failures({})
            self.diagram.board_number = ''
            self.diagram.states = {}
            self.diagram.draw()
            self.caption.set('เลือกบอร์ดเพื่อดูภาพอุปกรณ์')
            return
        board = self.entries[key]
        self.diagram.board_number = str(board['board_number'])
        records = {name:archive_entry(self.folder/key,board['results'].get(name,{})) for name in COMPONENTS}
        self.records = records
        self.show_failures({name:entry.get('failure_reason') or entry.get('reason','กำลังอ่านสาเหตุ…')
                            for name,entry in records.items() if entry.get('status') in ('FAIL','REVIEW')})
        self.diagram.states = {name:r.get('status','PENDING') for name,r in records.items()}
        self.diagram.draw()
        title = f"บอร์ด {board['board_number']} · {board['overall_status']}"
        self.caption.set(title+' · กำลังโหลดภาพอุปกรณ์…')
        cancel = self.cancel_tiles = threading.Event()
        ready = queue.Queue()
        def read_tiles():
            tiles = {}
            reasons={}
            for name,entry in records.items():
                if cancel.is_set():return
                if entry.get('status')=='FAIL':reasons[name]=failure_reason(entry)
                elif entry.get('status')=='REVIEW':reasons[name]=entry.get('reason','ยังตรวจไม่ครบทุกจุด')
                try:
                    tile = component_tile(name,entry)
                    if tile is not None:tiles[name] = tile
                except (OSError,ValueError,KeyError,TypeError,cv2.error):
                    continue
            ready.put((tiles,reasons))
        def display():
            if cancel.is_set():return
            try:tiles,reasons = ready.get_nowait()
            except queue.Empty:
                self.schedule(40,display)
                return
            self.diagram.tiles = tiles
            self.diagram.draw()
            self.show_failures(reasons)
            self.caption.set(f'{title} · ภาพ {len(tiles)}/{len(COMPONENTS)} อุปกรณ์ · คลิกบนผังเพื่อดูหลักฐาน')
        self.tile_worker=threading.Thread(target=read_tiles,daemon=True,name='board-evidence-photos')
        self.tile_worker.start()
        self.schedule(40,display)

    def open_part(self,name):
        if self.current_key not in self.entries:return
        self.app.show_evidence(name,archived_entry=self.records[name],
                               board_number=self.entries[self.current_key]['board_number'])
