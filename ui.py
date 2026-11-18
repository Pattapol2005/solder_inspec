"""Desktop dashboard: camera workspace, checklist, and on-demand settings."""
import tkinter as tk
from tkinter import ttk
from inspection import REGIONS
from board import COMPONENTS
from startup import PROJECT_TITLE_TH, PROJECT_TITLE_EN
from branding import ResponsiveLogo
from typography import fit_single_line
from board_layout import BoardLayout

COLORS = dict(bg='#edf2f5', card='#ffffff', ink='#192c3e', muted='#748497',
              accent='#087f73', good='#087f73', navy='#102333',
              line='#dbe4eb', red='#d14d57', amber='#b67b21')


def build_ui(app):
    c, root, family = COLORS, app.root, app.ui_font
    root.configure(bg=c['bg'])
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('.', font=(family, 10), bordercolor=c['line'], lightcolor=c['line'], darkcolor=c['line'])
    style.configure('TFrame',background=c['bg'])
    style.configure('TLabel',background=c['bg'],foreground=c['ink'])
    style.configure('TPanedwindow',background=c['bg'])
    for widget in ('TCombobox', 'TSpinbox'):
        style.configure(widget, foreground=c['ink'], arrowcolor=c['muted'], padding=6,
                        fieldbackground=c['card'], background=c['card'], bordercolor=c['line'])
    style.map('TCombobox', fieldbackground=[('readonly', 'white')], foreground=[('readonly', c['ink'])])
    style.configure('Treeview', font=(family, 10), rowheight=30, background='white',
                    fieldbackground='white', foreground=c['ink'], borderwidth=0)
    style.configure('Treeview.Heading', font=(family, 9, 'bold'), padding=8,
                    background='#f0f5f7', foreground=c['muted'], relief='flat')
    style.map('Treeview', background=[('selected', '#d8eee9')], foreground=[('selected', c['ink'])])
    style.configure('TNotebook', background='white', borderwidth=0)
    style.configure('TNotebook.Tab', padding=(12, 9), expand=(0,0,0,0), background='#eef3f6', foreground=c['muted'])
    style.map('TNotebook.Tab', background=[('selected', '#d8eee9')], foreground=[('selected', c['accent'])])
    style.configure('Section.TNotebook',background='white',borderwidth=0)
    style.layout('Section.TNotebook.Tab',[])
    style.configure('Vertical.TScrollbar', background='#dee8ef', troughcolor='#f0f5f7',
                    arrowcolor=c['muted'], bordercolor=c['line'])
    style.configure('Resource.Horizontal.TProgressbar', background=c['accent'],
                    troughcolor='#e7eff2', borderwidth=0, thickness=4)
    root.option_add('*TCombobox*Listbox.background', 'white')
    root.option_add('*TCombobox*Listbox.foreground', c['ink'])
    root.option_add('*TCombobox*Listbox.selectBackground', c['accent'])
    root.option_add('*TCombobox*Listbox.selectForeground', 'white')

    def label(parent, text='', size=10, color=None, bg=None, bold=False, **kw):
        return tk.Label(parent, text=text, font=(family, size, 'bold' if bold else 'normal'),
                        fg=color or c['ink'], bg=bg or c['card'], anchor='w', **kw)

    def button(parent, text, command, primary=False, **kw):
        return tk.Button(parent, text=text, command=command, bg=c['accent'] if primary else '#eef3f6',
                         fg='white' if primary else c['ink'], activebackground='#096b62' if primary else '#dee8ef',
                         activeforeground='white' if primary else c['ink'], disabledforeground='#a4b4bd',
                         relief='flat', bd=0, cursor='hand2', padx=kw.pop('padx',14), pady=kw.pop('pady',8),
                         font=(family, 10, 'bold'), **kw)

    def tab_navigation(parent,notebook):
        bar=tk.Frame(parent,bg=c['bg'])
        buttons={}
        for column,page in enumerate(notebook.tabs()):
            bar.columnconfigure(column,weight=1,uniform='tabs')
            control=button(bar,notebook.tab(page,'text'),lambda page=page:notebook.select(page),padx=6,pady=10)
            control.grid(row=0,column=column,sticky='ew',padx=2)
            buttons[page]=control
        def update(event=None):
            selected=notebook.select()
            for page,control in buttons.items():
                control.configure(bg=c['accent'] if page==selected else '#eef3f6',
                                  fg='white' if page==selected else c['ink'])
        notebook.bind('<<NotebookTabChanged>>',update,add='+')
        notebook.navigation_buttons=buttons
        update()
        return bar

    header = tk.Frame(root, bg=c['navy'], padx=24, pady=16)
    header.pack(fill='x')
    app.logo = ResponsiveLogo(header,c['navy'],280)
    app.logo.pack(side='left',padx=(0,24))
    app.logo.follow_width(header,.20,220,300,inset=48)
    brand = tk.Frame(header, bg=c['navy'])
    brand.pack(side='left', fill='x', expand=True)
    title_th = label(brand, PROJECT_TITLE_TH, 20, 'white', c['navy'], True, justify='left')
    fit_single_line(title_th, family, 20, bold=True)
    app.project_title_label = title_th
    title_th.pack(fill='x')
    title_en = label(brand, PROJECT_TITLE_EN, 9, '#a9bcc9', c['navy'], justify='left')
    title_en.pack(fill='x', pady=(5,0))
    def wrap_titles(event):
        title_en.configure(wraplength=max(100,event.width))
    brand.bind('<Configure>',wrap_titles)

    # Detailed controls remain available in a separate, reusable settings window.
    settings = tk.Toplevel(root)
    settings.withdraw()
    settings.title('ตั้งค่าโปรแกรม')
    settings.geometry('560x680')
    settings.minsize(540,650)
    settings.configure(bg=c['bg'])
    settings.protocol('WM_DELETE_WINDOW',settings.withdraw)
    app.settings_window = settings
    label(settings,'ตั้งค่าโปรแกรม',18,bg=c['bg'],bold=True).pack(anchor='w',padx=22,pady=(20,12))
    tabs = ttk.Notebook(settings)
    tabs.pack(fill='both',expand=True,padx=22,pady=(0,14))
    pages = [tk.Frame(tabs,bg='white',padx=20,pady=18) for _ in range(4)]
    camera_page, inspection_page, calibration_page, system_page = pages
    for page,title in zip(pages,('กล้อง','การตรวจ','ขา IC / ไดโอด','เครื่อง / AI')):tabs.add(page,text=title)
    button(settings,'เสร็จสิ้น',settings.withdraw,True).pack(anchor='e',padx=22,pady=(0,18))
    def open_settings():
        settings.deiconify()
        settings.lift()
    app.open_settings = open_settings
    app.settings_tabs=tabs
    app.settings_pages=dict(camera=camera_page,inspection=inspection_page,calibration=calibration_page,system=system_page)

    def field(parent,text,variable,choices=None,spin=None):
        caption = label(parent,text,10,c['muted'])
        caption.pack(anchor='w',pady=(8,5))
        if choices:
            widget = ttk.Combobox(parent,textvariable=variable,values=choices,state='readonly')
        else:
            widget = ttk.Spinbox(parent,textvariable=variable,from_=spin[0],to=spin[1],increment=spin[2])
        widget.pack(fill='x')
        widget.caption = caption
        return widget

    app.cam_index = tk.StringVar(value='0')
    app.resolution = tk.StringVar(value='640x480')
    app.backend = tk.StringVar(value='DirectShow')
    field(camera_page,'หมายเลขกล้อง USB',app.cam_index,spin=(0,20,1))
    field(camera_page,'ความละเอียดที่ร้องขอ',app.resolution,['640x480'])
    field(camera_page,'ระบบเชื่อมต่อกล้อง',app.backend,['Auto','DirectShow','MSMF'])
    label(camera_page,'เปลี่ยนค่าขณะหยุดกล้อง แล้วเชื่อมต่อใหม่',10,c['muted'],wraplength=390).pack(anchor='w',pady=16)
    controls = tk.Frame(camera_page,bg='white')
    controls.pack(fill='x')
    button(controls,'เชื่อมต่อกล้อง',app.connect,True).pack(side='left',padx=(0,8))
    button(controls,'หยุดกล้อง',app.disconnect).pack(side='left')

    label(inspection_page,'โหมดตรวจสอบ',10,c['muted']).pack(anchor='w',pady=(0,5))
    mode_controls=tk.Frame(inspection_page,bg='white');mode_controls.pack(fill='x',pady=(0,6))
    for text,value in [('อัตโนมัติ',True),('ตรวจเอง',False)]:
        tk.Radiobutton(mode_controls,text=text,variable=app.auto_mode,value=value,indicatoron=False,
                       command=app.change_scan_mode,bg='#eef3f6',fg=c['muted'],selectcolor='#d8eee9',
                       activebackground='#d8eee9',relief='flat',bd=0,padx=16,pady=6,font=(family,10)).pack(side='left',padx=(0,6))
    app.manual_settings = tk.Frame(inspection_page,bg='white')
    app.manual_settings.pack(fill='x')
    app.region = tk.StringVar(value=REGIONS[0])
    app.expected = tk.StringVar(value='0')
    field(app.manual_settings,'บริเวณบน PCB',app.region,REGIONS)
    field(app.manual_settings,'จำนวนจุดตรวจที่คาดหวัง',app.expected,spin=(0,500,1))
    app.auto_settings = tk.Frame(inspection_page,bg='white')
    app.scan_hold = tk.StringVar(value='0.6')
    app.scan_sharpness = tk.StringVar(value='18')
    app.locator_threshold = tk.StringVar(value=format(app.preferences['locator_confidence'],'.12g'))
    field(app.auto_settings,'Confidence ระบุอุปกรณ์ · 0.01–1.00',app.locator_threshold,spin=(.01,1,.01))
    app.threshold = tk.StringVar(value=format(app.preferences['inspection_confidence'],'.12g'))
    app.threshold_label = field(inspection_page,'Confidence ตรวจรอยบัดกรี · 0.01–1.00',app.threshold,spin=(.01,1,.01)).caption
    app.manual_hint = label(inspection_page,'ระบุขาที่เห็นในภาพ แล้วเลือกบริเวณตรวจ',10,c['muted'],wraplength=390,justify='left')
    app.manual_hint.pack(anchor='w',pady=(12,6))
    app.auto_hint = label(inspection_page,'ตรวจภาพล่าสุดต่อเนื่อง ไม่ต้องหยุดบอร์ด · จำค่า Confidence อัตโนมัติ',10,c['muted'],wraplength=390,justify='left')
    app.model_card = tk.Frame(inspection_page,bg='white')
    app.model_card.pack(fill='x',side='bottom')
    button(app.model_card,'ตั้งภาพอ้างอิงบอร์ดจริง',app.choose_reference).pack(fill='x',pady=(14,0))
    button(app.model_card,'ตั้งอ้างอิงจากภาพกล้องบริเวณนี้',app.choose_camera_reference).pack(fill='x',pady=(6,0))

    app.ic_component=tk.StringVar(value='U1')
    field(calibration_page,'IC ที่จะตั้งเลขขาจริง',app.ic_component,['U1','U4'])
    label(calibration_page,'U1: 14 ขา · U4: 28 ขา',10,c['muted']).pack(anchor='w',pady=(8,4))
    button(calibration_page,'ตั้งขา 1, 2 และขาฝั่งตรงข้าม',app.calibrate_ic,True).pack(fill='x',pady=(4,8))
    label(calibration_page,'ตั้ง 3 จุดบนภาพเดียว ไม่จำเป็นต้องเห็น IC ทั้งตัว\nขาที่ตรงข้ามขา 1 คือขา 14 หรือ 28\nเลื่อนกล้องโดยมีภาพซ้อนต่อกันเพื่อรักษาตำแหน่ง',10,c['muted'],wraplength=390,justify='left').pack(anchor='w',pady=(0,12))
    button(calibration_page,'กำหนดขั้ว + / - ที่ถูกต้องของไดโอด',app.calibrate_diode,True).pack(fill='x',pady=(4,8))
    label(calibration_page,'เลือกแพดตามขั้วบนบอร์ด โปรแกรมตรวจแถบคาดขั้วลบด้วย OpenCV และติดตามทิศเมื่อเลื่อนกล้อง',10,c['muted'],wraplength=390,justify='left').pack(anchor='w')
    app.calibration_status=tk.StringVar(value=app.calibration_summary())
    label(calibration_page,textvariable=app.calibration_status,size=9,color=c['accent'],wraplength=390,justify='left').pack(anchor='w',pady=(14,0))

    app.model_label = tk.StringVar(value='●  รอไฟล์ best_inspec.pt')
    app.model_detail = tk.StringVar(value='วาง best_inspec.pt และ best_locator.pt ข้าง app.py')
    label(system_page,'โมเดลตรวจรอยบัดกรี',12,bold=True).pack(anchor='w',pady=(0,10))
    label(system_page,size=11,color=c['accent'],textvariable=app.model_label).pack(anchor='w')
    label(system_page,size=10,color=c['muted'],textvariable=app.model_detail,wraplength=390,justify='left').pack(anchor='w',pady=(8,22))
    label(system_page,'ทรัพยากรเครื่อง',12,bold=True).pack(anchor='w')
    app.hardware_note = tk.StringVar(value='กำลังอ่านทรัพยากร · ค่า GPU ที่อ่านไม่ได้จะแสดง N/A')
    label(system_page,size=10,color=c['muted'],textvariable=app.hardware_note,wraplength=390,justify='left').pack(anchor='w',pady=10)
    label(system_page,'รายละเอียดการหาตำแหน่ง',12,bold=True).pack(anchor='w',pady=(8,4))
    label(system_page,textvariable=app.position_detail,size=9,color=c['muted'],wraplength=450,justify='left').pack(fill='x')
    button(system_page,'ทดสอบตำแหน่ง (ไม่ตรวจรอยบัดกรี)',app.locate_only).pack(fill='x',pady=(12,0))

    app.info = tk.StringVar(value='พร้อมเริ่มงาน · เชื่อมต่อกล้องหรือเปิดภาพ')

    from board_history import BoardHistory
    main_tabs=ttk.Notebook(root,style='Section.TNotebook')
    app.main_tabs=main_tabs
    main_tabs.pack(fill='both',expand=True)
    shell = tk.Frame(main_tabs,bg=c['bg'],padx=24,pady=8)
    main_tabs.add(shell,text='ตรวจสอบบอร์ด')
    app.board_history=BoardHistory(main_tabs,app)
    main_tabs.add(app.board_history,text='ประวัติบอร์ดที่ตรวจเสร็จ')
    tab_navigation(root,main_tabs).pack(fill='x',padx=24,pady=(10,0),before=main_tabs)
    def page_changed(event):
        if main_tabs.select()==str(app.board_history) and app.board_history.dirty:
            app.board_history.reload()
    main_tabs.bind('<<NotebookTabChanged>>',page_changed,add='+')
    app.board_progress = tk.StringVar(value=f'ครบ 0/{len(COMPONENTS)}')
    board_bar=tk.Frame(shell,bg='white',padx=10,pady=6)
    board_bar.pack(side='bottom',fill='x',pady=(8,0))
    label(board_bar,'เลขประจำบอร์ด',10).pack(side='left',padx=(0,8))
    app.board_number=tk.StringVar(value=app.checklist.session.get('board_number',''))
    app.board_number_entry=ttk.Entry(board_bar,textvariable=app.board_number,width=20)
    app.board_number_entry.pack(side='left',fill='x',expand=True,padx=(0,10))
    app.board_number_entry.bind('<FocusOut>',app.persist_board_number)
    button(board_bar,'รีเซ็ตการตรวจ',app.reset_board_inspection,pady=5).pack(side='left',padx=4)
    button(board_bar,'เริ่มบอร์ดใหม่',app.begin_new_board,pady=5).pack(side='left',padx=4)
    app.board_save_btn=button(board_bar,'บันทึกผลบอร์ดนี้',app.save_complete_board,True,state='disabled',pady=5)
    app.board_save_btn.pack(side='left',padx=(4,0))
    work = tk.Frame(shell,bg=c['bg'])
    work.pack(fill='both',expand=True)
    # Camera workspace with one results panel beside it.
    work.columnconfigure(0,weight=1,minsize=420)
    work.columnconfigure(1,weight=0,minsize=380)
    work.rowconfigure(0,weight=1)
    view = tk.Frame(work,bg='white')
    view.grid(row=0,column=0,sticky='nsew',padx=(0,10))
    app.view_panel = view
    view_header = tk.Frame(view,bg='white',padx=16,pady=10)
    view_header.pack(fill='x')
    app.camera_label = tk.StringVar(value='●  ไม่ได้เชื่อมต่อกล้อง')
    label(view_header,'ภาพกล้อง / ชิ้นงาน',11,bold=True).pack(side='left')
    selection = tk.Frame(view,bg='#f4f8fa',padx=14,pady=7)
    app.manual_selection = selection
    selection.pack(fill='x')
    app.component = tk.StringVar(value='ไม่ระบุ')
    label(selection,'อุปกรณ์',10,c['muted'],'#f4f8fa').pack(side='left',padx=(0,8))
    component_box = ttk.Combobox(selection,textvariable=app.component,values=['ไม่ระบุ',*COMPONENTS],width=8,state='readonly')
    component_box.pack(side='left')
    component_box.bind('<<ComboboxSelected>>',lambda event:app.select_component())
    label(selection,'ขาที่เห็น',10,c['muted'],'#f4f8fa').pack(side='left',padx=(16,8))
    app.joints = tk.StringVar()
    joint_box = ttk.Entry(selection,textvariable=app.joints,width=11)
    joint_box.pack(side='left')
    joint_box.bind('<KeyRelease>',app.update_joint_count)
    joint_box.bind('<FocusOut>',app.update_joint_count)
    label(selection,'เช่น 1-2,5',9,c['muted'],'#f4f8fa').pack(side='left',padx=8)
    app.manual_tools = tk.Frame(view,bg='white',padx=14,pady=6)
    app.manual_tools.pack(fill='x')
    app.roi_mode = tk.BooleanVar(value=True)
    tk.Checkbutton(app.manual_tools,text='ลากกรอบตรวจ',variable=app.roi_mode,bg='white',fg=c['accent'],
                   activebackground='white',bd=0).pack(side='left')
    app.rotation_label = tk.StringVar(value='หมุนกรอบ 0°')
    button(app.manual_tools,'ล้างกรอบ',app.clear_roi,pady=5).pack(side='right',padx=(6,0))
    button(app.manual_tools,'ดูภาพครอบ',app.preview_crop,pady=5).pack(side='right',padx=(6,0))
    button(app.manual_tools,'',app.rotate_roi,pady=5,textvariable=app.rotation_label).pack(side='right')
    app.auto_bar = tk.Frame(view,bg='#edf7f4',padx=10,pady=9)
    auto_note = label(app.auto_bar,size=10,color=c['accent'],bg='#edf7f4',textvariable=app.scan_status,justify='left')
    auto_note.pack(fill='x')
    app.auto_bar.bind('<Configure>',lambda event:auto_note.configure(wraplength=max(100,event.width-32)))

    app.viewport = tk.Frame(view,bg='white',width=640,height=480)
    app.canvas = tk.Canvas(app.viewport,bg='#101d2b',highlightthickness=0,width=640,height=480)
    app.viewport.bind('<Configure>',lambda event:app.render())
    app.canvas.bind('<Configure>',lambda event:app.render())
    app.canvas.bind('<ButtonPress-1>',app.roi_start)
    app.canvas.bind('<B1-Motion>',app.roi_drag)
    app.canvas.bind('<ButtonRelease-1>',app.roi_end)
    toolbar = tk.Frame(view,bg='white',padx=10,pady=8)
    toolbar.pack(side='bottom',fill='x')
    button(toolbar,'เชื่อมกล้อง',app.connect,True,padx=8,pady=6).pack(side='left',padx=(0,5))
    button(toolbar,'จับภาพ / ดูสด',app.freeze,padx=8,pady=6).pack(side='left',padx=(0,5))
    button(toolbar,'เปิดภาพ',app.open_image,padx=8,pady=6).pack(side='left')
    app.inspect_btn = button(toolbar,'ตรวจภาพนี้',app.inspect,True,state='disabled',padx=8,pady=6)
    app.inspect_btn.pack(side='right')
    more = tk.Menubutton(toolbar,text='⋯',bg='white',fg=c['muted'],relief='flat',padx=8,pady=6,cursor='hand2')
    more_menu = tk.Menu(more,tearoff=False)
    more_menu.add_command(label='ตรวจจุดที่ค้างใหม่',command=app.retry_scan)
    more_menu.add_command(label='บันทึกภาพต้นฉบับ',command=app.save_raw)
    more_menu.add_command(label='หยุดกล้อง',command=app.disconnect)
    more_menu.add_separator()
    more_menu.add_command(label='เปิดแผนที่บอร์ดแยกหน้าต่าง',command=app.show_board)
    more.configure(menu=more_menu)
    app.workspace_menu = more_menu
    more.pack(side='right',padx=6)

    app.source_label = tk.StringVar(value='ภาพจากกล้อง')
    app.image_size = tk.StringVar(value='— × — px')
    app.latency = tk.StringVar(value='—')
    app.overlay = tk.BooleanVar(value=True)
    tk.Checkbutton(inspection_page,text='แสดงกรอบและเลขขาบนภาพสด',variable=app.overlay,command=app.toggle_overlay,
                   bg='white',fg=c['muted'],activebackground='white',bd=0).pack(anchor='w',before=app.model_card,pady=(4,0))
    app.viewport.pack(fill='both',expand=True)

    results = tk.Frame(work,bg='white',padx=12,pady=12,width=320)
    results.grid(row=0,column=1,sticky='nsew')
    results.pack_propagate(False)
    app.results_panel = results
    result_section=tk.Frame(results,bg='white');result_section.pack(fill='x',pady=(0,10))
    result_heading=tk.Frame(result_section,bg='white');result_heading.pack(fill='x')
    app.settings_button=button(result_heading,'ตั้งค่าโปรแกรม',open_settings,padx=8,pady=5)
    app.settings_button.pack(side='right')
    label(result_heading,'ผลตรวจล่าสุด',12,bold=True).pack(side='left')
    result_header=tk.Frame(result_section,bg='white');result_header.pack(fill='x')
    app.component_status = tk.StringVar(value='รอจับภาพอุปกรณ์')
    app.status = tk.StringVar(value='STANDBY')
    app.status_widget = label(result_header,size=19,color=c['muted'],bold=True,textvariable=app.status)
    app.status_widget.pack(side='left')
    app.board_progress_label=label(result_section,textvariable=app.board_progress,size=9,color=c['muted'],wraplength=340,justify='left')
    app.board_progress_label.pack(fill='x',pady=(2,5))
    app.board_progress_label.bind('<Configure>',lambda event:app.board_progress_label.configure(wraplength=max(100,event.width-4)))
    label(result_section,textvariable=app.component_status,size=10,color=c['accent'],wraplength=340,justify='left').pack(fill='x')
    app.result_detail = tk.StringVar(value='เชื่อมต่อกล้อง หรือเปิดไฟล์ภาพ')
    label(result_section,textvariable=app.result_detail,size=9,color=c['muted'],wraplength=340,justify='left').pack(fill='x')
    app.diode_summary=tk.StringVar()
    app.diode_live_brief=tk.StringVar(value='ไดโอด · รอภาพ')
    label(results,'ผังบอร์ดและรายการตรวจ',12,bold=True).pack(fill='x',pady=(0,10))
    actions = tk.Frame(results,bg='white')
    actions.pack(side='bottom',fill='x')
    app.confirm_btn = button(actions,'ยืนยันเห็นครบจุด · ผล FAIL',app.confirm_visible_joints,state='disabled',pady=6)
    app.save_btn = button(actions,'บันทึกผล + CSV',app.save_result,state='disabled',pady=6)
    app.save_btn.pack(fill='x',pady=(6,0))
    notebook = ttk.Notebook(results,style='Section.TNotebook')
    notebook.pack(fill='both',expand=True)
    app.results_notebook = notebook
    table_page = tk.Frame(notebook,bg='white')
    board_page=tk.Frame(notebook,bg='white')
    notebook.add(board_page,text='ผังบอร์ด')
    notebook.add(table_page,text='รอยบัดกรี')
    pins_page=tk.Frame(notebook,bg='white',padx=8,pady=8)
    notebook.add(pins_page,text='ขา IC')
    diode_page=tk.Frame(notebook,bg='white',padx=8,pady=8)
    notebook.add(diode_page,text='ไดโอด')
    tab_navigation(results,notebook).pack(fill='x',pady=(0,8),before=notebook)
    app.diode_hint=tk.StringVar()
    label(diode_page,textvariable=app.diode_summary,size=9,color=c['accent'],justify='left',wraplength=330).pack(anchor='w',pady=(0,8))
    app.diode_details=tk.Text(diode_page,wrap='word',height=5,width=25,font=(family,10),bg='white',fg=c['ink'],bd=0,state='disabled')
    diode_scroll=ttk.Scrollbar(diode_page,command=app.diode_details.yview);diode_scroll.pack(side='right',fill='y')
    app.diode_details.configure(yscrollcommand=diode_scroll.set)
    app.diode_details.pack(fill='both',expand=True)
    pin_controls=tk.Frame(pins_page,bg='white');pin_controls.pack(fill='x')
    pin_selector=ttk.Combobox(pin_controls,textvariable=app.ic_component,values=['U1','U4'],state='readonly',width=6)
    pin_selector.pack(side='left')
    app.pin_progress=tk.StringVar(value='ยังไม่ตรวจขา IC')
    label(pins_page,textvariable=app.pin_progress,size=9,color=c['accent'],wraplength=330,justify='left').pack(anchor='w',pady=(4,6))
    pin_body=tk.Frame(pins_page,bg='white');pin_body.pack(fill='both',expand=True)
    app.pin_table=ttk.Treeview(pin_body,columns=('pin','state'),show='headings',height=8)
    app.pin_table.heading('pin',text='เลขขาจริง');app.pin_table.heading('state',text='ผลตรวจ')
    app.pin_table.column('pin',width=110,anchor='center');app.pin_table.column('state',width=140,anchor='center')
    app.pin_table.pack(side='left',fill='both',expand=True)
    pin_scroll=ttk.Scrollbar(pin_body,command=app.pin_table.yview);pin_scroll.pack(side='right',fill='y',before=app.pin_table)
    app.pin_table.configure(yscrollcommand=pin_scroll.set)
    for state,color in [('PASS',c['good']),('FAIL',c['red']),('REVIEW',c['amber'])]:app.pin_table.tag_configure(state,foreground=color)
    legend=tk.Frame(board_page,bg='white');legend.pack(side='bottom',fill='x',padx=6,pady=5)
    for i,(title,color) in enumerate([('รอตรวจ','#748497'),('พบในภาพ','#238cb5'),('ผ่าน',c['good']),('ไม่ผ่าน',c['red']),('ตรวจไม่ครบ',c['amber'])]):
        label(legend,'● '+title,9,color).grid(row=i//3,column=i%3,sticky='w',padx=(0,10),pady=1)
    diagram_frame=tk.Frame(board_page,bg='white')
    diagram_frame.pack(fill='both',expand=True,padx=4,pady=4)
    app.board_layout=BoardLayout(diagram_frame,app)
    app.board_layout.pack(side='left',fill='both',expand=True)
    diagram_scroll=ttk.Scrollbar(diagram_frame,command=app.board_layout.yview)
    diagram_scroll.pack(side='right',fill='y',before=app.board_layout)
    app.board_layout.configure(yscrollcommand=diagram_scroll.set)
    # Counts belong with detection details, leaving the workspace uncluttered.
    metrics_section = tk.Frame(table_page,bg='white',padx=10,pady=8);metrics_section.pack(fill='x')
    metrics=tk.Frame(metrics_section,bg='white');metrics.pack(fill='x')
    app.counts = []
    for i,(name,color) in enumerate([('ปกติ',c['good']),('เชื่อมกัน',c['red']),('ตะกั่วมาก',c['amber']),('ขาดหาย','#7567a7')]):
        cell = tk.Frame(metrics,bg='white')
        cell.grid(row=i//2,column=i%2,sticky='ew',padx=(0,10),pady=5)
        metrics.columnconfigure(i%2,weight=1)
        label(cell,name,9,c['muted']).pack(side='left')
        value=tk.StringVar(value='—');app.counts.append(value)
        label(cell,size=15,color=color,bold=True,textvariable=value).pack(side='right')
    table_frame=tk.Frame(table_page,bg='white')
    table_frame.pack(fill='both',expand=True)
    app.table = ttk.Treeview(table_frame,columns=('type','conf'),show='headings',height=6)
    for column,title,width in [('type','ประเภท',170),('conf','คะแนน',72)]:
        app.table.heading(column,text=title)
        app.table.column(column,width=width,stretch=column=='type',anchor='center')
    app.table.pack(side='left',fill='both',expand=True)
    app.table.tag_configure('good',foreground=c['good'])
    app.table.tag_configure('defect',foreground=c['red'])
    scroll=ttk.Scrollbar(table_frame,command=app.table.yview)
    scroll.pack(side='right',fill='y',before=app.table)
    app.table.configure(yscrollcommand=scroll.set)

    # Restore the full-width resource strip and machine specifications.
    resources=tk.Frame(root,bg='white',padx=24,pady=8)
    resources.pack(side='bottom',fill='x',before=main_tabs)
    app.resource_values,app.resource_bars={},{}
    for i,(key,title) in enumerate([('cpu','CPU'),('ram','RAM'),('gpu','GPU 0'),('vram','VRAM')]):
        resources.columnconfigure(i,weight=1,uniform='resource')
        cell=tk.Frame(resources,bg='white');cell.grid(row=0,column=i,sticky='ew',padx=(0,16))
        label(cell,title,9,c['muted']).pack(side='left')
        value=tk.StringVar(value='…');app.resource_values[key]=value
        label(cell,textvariable=value,size=10,bold=True).pack(side='right')
        bar=ttk.Progressbar(resources,maximum=100,style='Resource.Horizontal.TProgressbar')
        bar.grid(row=1,column=i,sticky='ew',padx=(0,16),pady=(3,6));app.resource_bars[key]=bar
    import platform, psutil
    cpu=app.runtime_info.get('cpu_name') or platform.processor() or 'CPU'
    gpu=' / '.join(app.runtime_info.get('gpu_names',[])) or app.runtime_info.get('device_name','GPU: รออ่านข้อมูล')
    app.resource_details = {}
    for index, (key, description) in enumerate((('cpu', cpu), ('ram', 'หน่วยความจำเครื่อง'),
                                                ('gpu', gpu), ('vram', 'หน่วยความจำกราฟิก'))):
        variable = tk.StringVar(value=description)
        app.resource_details[key] = variable
        caption = label(resources, textvariable=variable, size=9, color=c['muted'], justify='left')
        caption.grid(row=2, column=index, sticky='ew', padx=(0, 16))
        caption.bind('<Configure>', lambda event, widget=caption: widget.configure(wraplength=max(80, event.width-4)))
    app.resource_panel=resources
    viewer_info=tk.Frame(view,bg='white',padx=16,pady=5)
    viewer_info.pack(side='bottom',fill='x',before=toolbar)
    label(viewer_info,textvariable=app.image_size,size=9,color=c['muted']).pack(side='left')
    label(viewer_info,textvariable=app.latency,size=9,color=c['muted']).pack(side='right')
    label(view_header,textvariable=app.camera_label,size=9,color=c['muted']).pack(side='right')
    footer=tk.Frame(root,bg=c['bg'],padx=24,pady=4)
    footer.pack(side='bottom',fill='x',before=main_tabs)
    label(footer,textvariable=app.info,size=9,color=c['muted'],bg=c['bg']).pack(fill='x')
    app.workspace=work
    work.bind('<Configure>',lambda event:work.columnconfigure(1,minsize=max(380,min(480,round(event.width*.28)))))
