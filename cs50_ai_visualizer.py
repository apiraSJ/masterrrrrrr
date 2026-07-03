"""
CS50 AI Optimization & Constraint Satisfaction Visualizer
============================================================
TAB 1 — Pathfinding / Optimization algorithms:
    * A* Search                 (g + h, optimal & complete)
    * Breadth-First Search      (guarantees fewest steps, uniform cost)
    * Greedy Best-First Search  (uses h only, fast but NOT optimal)
    * Hill Climbing             (greedy local search, can get stuck)
    * Simulated Annealing       (local search that can escape local minima)

TAB 2 — Constraint Satisfaction Problem (CSP) solver:
    Classic map-colouring problem (Australia regions), solved with
    Backtracking Search + Forward Checking + MRV heuristic.
    You can adjust:
      * Domain size (number of colours available, 2-4)
      * Whether SOFT constraints (colour preferences) are honoured
        in addition to the HARD constraints (neighbouring regions
        must differ).
"""

import tkinter as tk
from tkinter import ttk
import random
import math
import collections
import heapq


# ──────────────────────────────────────────────
#  CONSTANTS & COLOUR PALETTE
# ──────────────────────────────────────────────
GRID_SIZE   = 15
CELL_PX     = 42
PAD         = 3

C_BG        = "#1a1a2e"
C_PANEL     = "#16213e"
C_ACCENT    = "#0f3460"
C_GRID_LINE = "#0d2137"

C_EMPTY     = "#13294d"
C_WALL      = "#e94560"
C_START     = "#4ecca3"
C_GOAL      = "#f9ca24"
C_VISITED   = "#1a4a6e"
C_FRONTIER  = "#2980b9"
C_PATH      = "#a29bfe"
C_CURRENT   = "#fd79a8"
C_BACKTRACK = "#e17055"
C_TEXT_MAIN = "#eaeaea"
C_TEXT_DIM  = "#74b9ff"
C_COST_TXT  = "#ffffff"

EMPTY = 0; WALL = 1; START = 2; GOAL = 3

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# CSP tab palette
C_CSP_NODE_DEFAULT  = "#2d2d44"
C_CSP_NODE_BORDER   = "#4a4a68"
C_CSP_NODE_CURRENT  = "#f9ca24"
C_CSP_EDGE          = "#4a4a68"
C_CSP_EDGE_CONFLICT = "#e94560"
CSP_COLOR_PALETTE = [
    ("#e94560", "Red"),
    ("#4ecca3", "Green"),
    ("#f9ca24", "Yellow"),
    ("#6c5ce7", "Purple"),
]


# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def neighbors(grid, r, c):
    result = []
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and grid[nr][nc] != WALL:
            result.append((nr, nc))
    return result

def in_bounds(r, c):
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE

def is_passable(grid, r, c):
    return in_bounds(r, c) and grid[r][c] != WALL


# ──────────────────────────────────────────────
#  SCROLLABLE CONTAINER
#  Wraps a tab's content in a canvas + scrollbars so that on smaller
#  screens/windows nothing is ever cut off — the user can simply scroll.
# ──────────────────────────────────────────────
class ScrollableFrame(tk.Frame):
    def __init__(self, parent, bg):
        super().__init__(parent, bg=bg)

        self._canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        vsb = tk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        hsb = tk.Scrollbar(self, orient="horizontal", command=self._canvas.xview)

        # `body` is where real content should be packed/gridded into.
        self.body = tk.Frame(self._canvas, bg=bg)
        self._window = self._canvas.create_window((0, 0), window=self.body, anchor="nw")

        self.body.bind("<Configure>",
                       lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self._canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Mouse-wheel scrolling only while the pointer is over this canvas,
        # so multiple scrollable areas (e.g. two tabs) don't fight for it.
        def _on_wheel(event):
            delta = event.delta
            if delta == 0:
                return
            self._canvas.yview_scroll(int(-1 * (delta / 120)), "units")

        def _bind_wheel(_):
            self._canvas.bind_all("<MouseWheel>", _on_wheel)

        def _unbind_wheel(_):
            self._canvas.unbind_all("<MouseWheel>")

        self._canvas.bind("<Enter>", _bind_wheel)
        self._canvas.bind("<Leave>", _unbind_wheel)


# ──────────────────────────────────────────────
#  PATHFINDING TAB APPLICATION
# ──────────────────────────────────────────────
class PathfindingApp:
    """Everything for TAB 1. `parent` is any Tk container (Frame/Notebook page)."""

    def __init__(self, parent):
        self.parent = parent

        self.grid       = [[EMPTY] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.start      = None
        self.goal       = None
        self.place_mode = tk.StringVar(value="wall")

        # Animation state
        self.running    = False
        self.paused     = False
        self.after_id   = None
        self.anim_steps = []
        self.anim_index = 0
        self.overlay    = {}
        self.cost_map   = {}
        self.last_algo  = None

        self._build_ui()
        self._draw_grid()

    # ── UI CONSTRUCTION ──────────────────────────────────
    def _build_ui(self):
        canvas_frame = tk.Frame(self.parent, bg=C_BG)
        canvas_frame.grid(row=0, column=0, padx=(16, 8), pady=16, sticky="n")

        canvas_w = GRID_SIZE * CELL_PX + 2
        canvas_h = GRID_SIZE * CELL_PX + 2
        self.canvas = tk.Canvas(canvas_frame, width=canvas_w, height=canvas_h,
                                bg=C_GRID_LINE, highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>",  self._on_left_click)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<Button-3>",  self._on_right_click)

        ctrl = tk.Frame(self.parent, bg=C_PANEL, width=320)
        ctrl.grid(row=0, column=1, padx=(8, 16), pady=16, sticky="ns")
        ctrl.grid_propagate(False)

        pad = dict(padx=12, pady=4)

        # Title
        tk.Label(ctrl, text="CS50 AI", bg=C_PANEL, fg=C_ACCENT,
                 font=("Helvetica", 11, "bold")).pack(**pad, anchor="w")
        tk.Label(ctrl, text="Optimization Visualizer", bg=C_PANEL, fg=C_TEXT_MAIN,
                 font=("Helvetica", 14, "bold")).pack(padx=12, pady=(0, 2), anchor="w")
        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=5)

        # ── Place Mode ──
        tk.Label(ctrl, text="PLACE MODE", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9, "bold")).pack(**pad, anchor="w")
        mode_frame = tk.Frame(ctrl, bg=C_PANEL)
        mode_frame.pack(padx=12, pady=2, fill="x")
        for text, val, color in [("Start (S)", "start", C_START),
                                   ("Goal (G)",  "goal",  C_GOAL),
                                   ("Wall (X)",  "wall",  C_WALL)]:
            tk.Radiobutton(mode_frame, text=text, variable=self.place_mode,
                           value=val, bg=C_PANEL, fg=color, selectcolor=C_ACCENT,
                           activebackground=C_PANEL, activeforeground=color,
                           font=("Helvetica", 10, "bold"), cursor="hand2"
                           ).pack(anchor="w", pady=1)

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=5)

        # ── Algorithm ──
        tk.Label(ctrl, text="ALGORITHM", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9, "bold")).pack(**pad, anchor="w")
        self.algo_var = tk.StringVar(value="A* Search")
        ttk.Combobox(ctrl, textvariable=self.algo_var, state="readonly",
                     font=("Helvetica", 10),
                     values=["A* Search",
                             "Breadth-First Search (BFS)",
                             "Greedy Best-First Search",
                             "Hill Climbing",
                             "Simulated Annealing"]
                     ).pack(padx=12, pady=4, fill="x")

        self.algo_desc_var = tk.StringVar(value="")
        tk.Label(ctrl, textvariable=self.algo_desc_var, bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 8), wraplength=280, justify="left"
                 ).pack(padx=12, pady=(0, 2), anchor="w")
        self.algo_var.trace_add("write", lambda *_: self._update_algo_desc())
        self._update_algo_desc()

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=5)

        # ── Speed ──
        tk.Label(ctrl, text="ANIMATION SPEED", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9, "bold")).pack(**pad, anchor="w")
        spd_frame = tk.Frame(ctrl, bg=C_PANEL)
        spd_frame.pack(padx=12, pady=2, fill="x")
        tk.Label(spd_frame, text="Fast", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 8)).pack(side="left")
        self.speed_var = tk.IntVar(value=150)
        tk.Scale(spd_frame, variable=self.speed_var, from_=20, to=800,
                 orient="horizontal", bg=C_PANEL, fg=C_TEXT_MAIN,
                 troughcolor=C_ACCENT, highlightthickness=0,
                 showvalue=False).pack(side="left", fill="x", expand=True)
        tk.Label(spd_frame, text="Slow", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 8)).pack(side="left")

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=5)

        # ── Primary Buttons ──
        btn_cfg = dict(font=("Helvetica", 11, "bold"), bd=0, cursor="hand2",
                       pady=7, relief="flat")

        self.run_btn = tk.Button(ctrl, text="▶  RUN",
                                 bg="#00b894", fg="white",
                                 command=self._run, **btn_cfg)
        self.run_btn.pack(padx=12, pady=3, fill="x")

        self.pause_btn = tk.Button(ctrl, text="⏸  PAUSE",
                                   bg="#6c5ce7", fg="white",
                                   command=self._toggle_pause,
                                   state="disabled", **btn_cfg)
        self.pause_btn.pack(padx=12, pady=3, fill="x")

        self.stop_btn = tk.Button(ctrl, text="■  STOP",
                                  bg="#d63031", fg="white",
                                  command=self._stop,
                                  state="disabled", **btn_cfg)
        self.stop_btn.pack(padx=12, pady=3, fill="x")

        # ── Step Navigation ──
        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=5)
        tk.Label(ctrl, text="STEP NAVIGATION", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9, "bold")).pack(**pad, anchor="w")

        nav_frame = tk.Frame(ctrl, bg=C_PANEL)
        nav_frame.pack(padx=12, pady=3, fill="x")
        nav_btn = dict(font=("Helvetica", 10, "bold"), bd=0, cursor="hand2",
                       pady=6, relief="flat", bg=C_ACCENT, fg=C_TEXT_MAIN)
        self.prev_btn = tk.Button(nav_frame, text="◀  PREV",
                                  command=self._step_back,
                                  state="disabled", **nav_btn)
        self.prev_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.next_btn = tk.Button(nav_frame, text="NEXT  ▶",
                                  command=self._step_forward,
                                  state="disabled", **nav_btn)
        self.next_btn.pack(side="left", expand=True, fill="x")

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=5)

        # ── Utility Buttons ──
        tk.Button(ctrl, text="🎲  RANDOM MAZE",
                  bg="#fdcb6e", fg="#2d3436",
                  command=self._random_maze, **btn_cfg
                  ).pack(padx=12, pady=3, fill="x")
        tk.Button(ctrl, text="↺  RESET GRID",
                  bg=C_ACCENT, fg=C_TEXT_MAIN,
                  command=self._reset, **btn_cfg
                  ).pack(padx=12, pady=3, fill="x")

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=5)

        # ── Legend ──
        tk.Label(ctrl, text="LEGEND", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9, "bold")).pack(**pad, anchor="w")
        legend_items = [
            (C_START,    "Start"),
            (C_GOAL,     "Goal"),
            (C_WALL,     "Wall"),
            (C_VISITED,  "Visited"),
            (C_CURRENT,  "Current"),
            (C_PATH,     "Final Path"),
            (C_FRONTIER, "Frontier"),
            (C_BACKTRACK,"Stuck / Local Min"),
        ]
        leg_frame = tk.Frame(ctrl, bg=C_PANEL)
        leg_frame.pack(padx=12, pady=2, fill="x")
        for i, (color, label) in enumerate(legend_items):
            row_f = tk.Frame(leg_frame, bg=C_PANEL)
            row_f.grid(row=i // 2, column=i % 2, sticky="w", padx=2, pady=1)
            tk.Label(row_f, bg=color, width=2, height=1).pack(side="left", padx=(0, 4))
            tk.Label(row_f, text=label, bg=C_PANEL, fg=C_TEXT_MAIN,
                     font=("Helvetica", 8)).pack(side="left")

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=5)

        # ── Status & Step Counter ──
        tk.Label(ctrl, text="STATUS", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9, "bold")).pack(**pad, anchor="w")
        self.status_var = tk.StringVar(value="Ready. Place Start (S) and Goal (G), add walls, then RUN.")
        tk.Label(ctrl, textvariable=self.status_var,
                 bg=C_ACCENT, fg=C_TEXT_MAIN,
                 font=("Helvetica", 9), wraplength=280,
                 justify="left", anchor="nw"
                 ).pack(padx=12, pady=4, fill="x", ipadx=8, ipady=8)

        self.step_var = tk.StringVar(value="Step: —")
        tk.Label(ctrl, textvariable=self.step_var, bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9)).pack(padx=12, anchor="w")

        # ── Results ──
        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=5)
        tk.Label(ctrl, text="RESULTS", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9, "bold")).pack(**pad, anchor="w")

        results_box = tk.Frame(ctrl, bg=C_ACCENT)
        results_box.pack(padx=12, pady=4, fill="x")

        self.cost_var  = tk.StringVar(value="Total Cost Used: —")
        self.dist_var  = tk.StringVar(value="Distance Traveled: —")
        self.optim_var = tk.StringVar(value="")

        lbl_cost = tk.Label(results_box, textvariable=self.cost_var,
                            bg=C_ACCENT, fg=C_TEXT_MAIN,
                            font=("Helvetica", 10, "bold"),
                            anchor="w", justify="left")
        lbl_cost.pack(fill="x", padx=8, pady=(8, 2))

        lbl_dist = tk.Label(results_box, textvariable=self.dist_var,
                            bg=C_ACCENT, fg=C_TEXT_MAIN,
                            font=("Helvetica", 9),
                            anchor="w", justify="left", wraplength=280)
        lbl_dist.pack(fill="x", padx=8, pady=2)

        lbl_optim = tk.Label(results_box, textvariable=self.optim_var,
                             bg=C_ACCENT, fg="#a8e6cf",
                             font=("Helvetica", 8, "bold"),
                             anchor="w", justify="left", wraplength=280)
        lbl_optim.pack(fill="x", padx=8, pady=(2, 8))

    def _update_algo_desc(self):
        descs = {
            "A* Search":
                "f(n) = g(n) + h(n). ใช้ต้นทุนจริง (g) + ค่าประมาณ Manhattan (h). "
                "รับประกันเส้นทางที่สั้นที่สุด (Optimal) เมื่อ h เป็น Admissible/Consistent.",
            "Breadth-First Search (BFS)":
                "สำรวจโหนดที่ตื้นที่สุดก่อน (ทีละระดับ). รับประกันจำนวนก้าวน้อยที่สุด "
                "เมื่อทุกก้าวมีต้นทุนเท่ากัน แต่สำรวจโหนดเยอะกว่า A*.",
            "Greedy Best-First Search":
                "ใช้เฉพาะค่า h(n) เพื่อพุ่งเข้าเป้าหมายให้เร็วที่สุด. เร็วและสำรวจน้อย "
                "แต่ไม่รับประกันเส้นทางสั้นที่สุด (Not optimal).",
            "Hill Climbing":
                "เดินไปยังเพื่อนบ้านที่ h(n) ดีขึ้นเสมอ (Greedy local search). "
                "ปรับปรุงคำตอบไปเรื่อยๆ แต่ติด Local Minimum ได้ง่าย.",
            "Simulated Annealing":
                "คล้าย Hill Climbing แต่ยอมรับทางที่แย่ลงได้ด้วยความน่าจะเป็นที่ลดลงตาม "
                "อุณหภูมิ (T) ช่วยหนีจาก Local Minimum ในปัญหาที่ซับซ้อน.",
        }
        self.algo_desc_var.set(descs.get(self.algo_var.get(), ""))

    # ── GRID DRAWING ─────────────────────────────────────
    def _cell_coords(self, r, c):
        x0 = c * CELL_PX + PAD
        y0 = r * CELL_PX + PAD
        x1 = x0 + CELL_PX - PAD * 2
        y1 = y0 + CELL_PX - PAD * 2
        return x0, y0, x1, y1

    def _draw_grid(self):
        self.canvas.delete("all")
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                self._draw_cell(r, c)

    def _cell_color(self, r, c):
        s = self.grid[r][c]
        if s == START: return C_START
        if s == GOAL:  return C_GOAL
        if s == WALL:  return C_WALL
        if (r, c) in self.overlay:
            return self.overlay[(r, c)]
        return C_EMPTY

    def _draw_cell(self, r, c):
        x0, y0, x1, y1 = self._cell_coords(r, c)
        color = self._cell_color(r, c)
        tag = f"cell_{r}_{c}"
        self.canvas.delete(tag)
        self.canvas.create_rectangle(x0, y0, x1, y1,
                                     fill=color, outline=C_GRID_LINE,
                                     width=1, tags=tag)
        state = self.grid[r][c]
        if state == START:
            self.canvas.create_text((x0 + x1) // 2, (y0 + y1) // 2,
                                    text="S", fill=C_BG,
                                    font=("Helvetica", int(CELL_PX * 0.38), "bold"),
                                    tags=tag)
        elif state == GOAL:
            self.canvas.create_text((x0 + x1) // 2, (y0 + y1) // 2,
                                    text="G", fill=C_BG,
                                    font=("Helvetica", int(CELL_PX * 0.38), "bold"),
                                    tags=tag)
        elif state == WALL:
            self.canvas.create_text((x0 + x1) // 2, (y0 + y1) // 2,
                                    text="✕", fill="white",
                                    font=("Helvetica", int(CELL_PX * 0.32), "bold"),
                                    tags=tag)
        elif (r, c) in self.cost_map:
            label = self.cost_map[(r, c)]
            lines = str(label).split("\n")
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            if len(lines) == 1:
                self.canvas.create_text(cx, cy,
                                        text=lines[0],
                                        fill=C_TEXT_MAIN,
                                        font=("Helvetica", int(CELL_PX * 0.24)),
                                        tags=tag)
            else:
                fsize = int(CELL_PX * 0.22)
                gap   = fsize + 2
                self.canvas.create_text(cx, cy - gap // 2,
                                        text=lines[0],
                                        fill="#a8e6cf",
                                        font=("Helvetica", fsize),
                                        tags=tag)
                self.canvas.create_text(cx, cy + gap // 2,
                                        text=lines[1],
                                        fill="#ffd3b6",
                                        font=("Helvetica", fsize),
                                        tags=tag)

    def _redraw_overlay(self):
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                self._draw_cell(r, c)

    # ── MOUSE INTERACTION ────────────────────────────────
    def _rc_from_event(self, event):
        c = event.x // CELL_PX
        r = event.y // CELL_PX
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
            return r, c
        return None, None

    def _place(self, r, c):
        mode = self.place_mode.get()
        if mode == "start":
            if self.start:
                sr, sc = self.start
                self.grid[sr][sc] = EMPTY
                self._draw_cell(sr, sc)
            self.start = (r, c)
            self.grid[r][c] = START
        elif mode == "goal":
            if self.goal:
                gr, gc = self.goal
                self.grid[gr][gc] = EMPTY
                self._draw_cell(gr, gc)
            self.goal = (r, c)
            self.grid[r][c] = GOAL
        else:
            if self.grid[r][c] == WALL:
                self.grid[r][c] = EMPTY
            elif self.grid[r][c] == EMPTY:
                self.grid[r][c] = WALL
        self._draw_cell(r, c)

    def _on_left_click(self, event):
        if self.running: return
        r, c = self._rc_from_event(event)
        if r is not None:
            self._place(r, c)

    def _on_left_drag(self, event):
        if self.running: return
        r, c = self._rc_from_event(event)
        if r is not None and self.place_mode.get() == "wall":
            if self.grid[r][c] == EMPTY:
                self.grid[r][c] = WALL
                self._draw_cell(r, c)

    def _on_right_click(self, event):
        if self.running: return
        r, c = self._rc_from_event(event)
        if r is not None and self.grid[r][c] == WALL:
            self.grid[r][c] = EMPTY
            self._draw_cell(r, c)

    # ── RANDOM MAZE ──────────────────────────────────────
    def _random_maze(self):
        self._stop()
        self.overlay    = {}
        self.cost_map   = {}
        self.anim_steps = []
        self.anim_index = 0
        self.last_algo  = None
        self._reset_results()

        N = GRID_SIZE
        g = [[WALL] * N for _ in range(N)]

        LOGICAL = (N - 1) // 2
        visited = [[False] * LOGICAL for _ in range(LOGICAL)]

        def carve(lr, lc):
            visited[lr][lc] = True
            gr, gc = lr * 2 + 1, lc * 2 + 1
            g[gr][gc] = EMPTY
            dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            random.shuffle(dirs)
            for dr, dc in dirs:
                nr, nc = lr + dr, lc + dc
                if 0 <= nr < LOGICAL and 0 <= nc < LOGICAL and not visited[nr][nc]:
                    wr, wc = gr + dr, gc + dc
                    g[wr][wc] = EMPTY
                    carve(nr, nc)

        carve(0, 0)

        num_shortcuts = random.randint(2, 3)
        shortcuts_added = 0
        attempts = 0
        while shortcuts_added < num_shortcuts and attempts < 200:
            attempts += 1
            r = random.randint(1, N - 2)
            c = random.randint(1, N - 2)
            if g[r][c] == WALL:
                if g[r][c - 1] == EMPTY and g[r][c + 1] == EMPTY:
                    g[r][c] = EMPTY
                    shortcuts_added += 1
                elif g[r - 1][c] == EMPTY and g[r + 1][c] == EMPTY:
                    g[r][c] = EMPTY
                    shortcuts_added += 1

        def pick_empty(r_lo, r_hi, c_lo, c_hi):
            cells = [(r, c) for r in range(r_lo, r_hi)
                             for c in range(c_lo, c_hi)
                             if g[r][c] == EMPTY]
            return random.choice(cells) if cells else None

        s  = pick_empty(1, N // 3 + 1,      1, N // 3 + 1)
        gl = pick_empty(N * 2 // 3, N - 1,  N * 2 // 3, N - 1)

        if s is None:
            for r in range(N):
                for c in range(N):
                    if g[r][c] == EMPTY:
                        s = (r, c); break
                if s: break
        if gl is None:
            for r in range(N - 1, -1, -1):
                for c in range(N - 1, -1, -1):
                    if g[r][c] == EMPTY and (r, c) != s:
                        gl = (r, c); break
                if gl: break

        g[s[0]][s[1]]   = START
        g[gl[0]][gl[1]] = GOAL

        self.grid  = g
        self.start = s
        self.goal  = gl

        self._draw_grid()
        self.status_var.set(
            f"Maze generated ({num_shortcuts} shortcuts carved). Click RUN to solve.")
        self.step_var.set("Step: —")

    # ── CONTROLS ─────────────────────────────────────────
    def _run(self):
        if not self.start or not self.goal:
            self.status_var.set("⚠  Place both Start (S) and Goal (G) first.")
            return

        self.running    = True
        self.paused     = False
        self.overlay    = {}
        self.cost_map   = {}
        self.anim_steps = []
        self.anim_index = 0
        self._reset_results()

        self.run_btn.config(state="disabled")
        self.pause_btn.config(state="normal", text="⏸  PAUSE", bg="#6c5ce7")
        self.stop_btn.config(state="normal")
        self.prev_btn.config(state="disabled")
        self.next_btn.config(state="disabled")

        algo = self.algo_var.get()
        if   algo == "A* Search":                    algo_obj = AStarSearch(self.grid, self.start, self.goal)
        elif algo == "Breadth-First Search (BFS)":    algo_obj = BFSSearch(self.grid, self.start, self.goal)
        elif algo == "Greedy Best-First Search":      algo_obj = GreedyBestFirstSearch(self.grid, self.start, self.goal)
        elif algo == "Hill Climbing":                 algo_obj = HillClimbing(self.grid, self.start, self.goal)
        elif algo == "Simulated Annealing":           algo_obj = SimulatedAnnealing(self.grid, self.start, self.goal)

        self.anim_steps = algo_obj.solve()
        self.last_algo  = algo_obj

        self.anim_index = 0
        self._animate()

    def _animate(self):
        if not self.running or self.paused:
            return
        if self.anim_index >= len(self.anim_steps):
            self._finish()
            return
        self._apply_step(self.anim_index)
        self.anim_index += 1
        self.after_id = self.parent.after(self.speed_var.get(), self._animate)

    def _apply_step(self, idx):
        overlay, cost_map, status = self.anim_steps[idx]
        self.overlay  = overlay
        self.cost_map = cost_map
        self._redraw_overlay()
        self.status_var.set(status)
        self.step_var.set(f"Step: {idx + 1} / {len(self.anim_steps)}")

    def _finish(self):
        self.running = False
        self.paused  = False
        self.run_btn.config(state="normal")
        self.pause_btn.config(state="disabled", text="⏸  PAUSE")
        self.stop_btn.config(state="disabled")
        self._update_nav_buttons()
        self._show_results()

    def _toggle_pause(self):
        if not self.running:
            return
        self.paused = not self.paused
        if self.paused:
            if self.after_id:
                self.parent.after_cancel(self.after_id)
                self.after_id = None
            self.pause_btn.config(text="▶  RESUME", bg="#00b894")
            self._update_nav_buttons()
            self.status_var.set("⏸  Paused. Use ◀ PREV / NEXT ▶ to navigate steps.")
        else:
            self.pause_btn.config(text="⏸  PAUSE", bg="#6c5ce7")
            self.prev_btn.config(state="disabled")
            self.next_btn.config(state="disabled")
            self._animate()

    def _stop(self):
        was_running = self.running
        if self.after_id:
            self.parent.after_cancel(self.after_id)
            self.after_id = None
        self.running = False
        self.paused  = False
        self.run_btn.config(state="normal")
        self.pause_btn.config(state="disabled", text="⏸  PAUSE", bg="#6c5ce7")
        self.stop_btn.config(state="disabled")
        self._update_nav_buttons()
        self.status_var.set("⏹  Stopped.")
        if was_running:
            self._show_results()

    def _step_back(self):
        if self.anim_index > 1:
            self.anim_index -= 1
            self._apply_step(self.anim_index - 1)
        self._update_nav_buttons()

    def _step_forward(self):
        if self.anim_index < len(self.anim_steps):
            self._apply_step(self.anim_index)
            self.anim_index += 1
        self._update_nav_buttons()

    def _update_nav_buttons(self):
        can_nav = (not self.running or self.paused) and len(self.anim_steps) > 0
        self.prev_btn.config(state="normal" if (can_nav and self.anim_index > 1) else "disabled")
        self.next_btn.config(state="normal" if (can_nav and self.anim_index < len(self.anim_steps)) else "disabled")

    def _reset_results(self):
        self.cost_var.set("Total Cost Used: —")
        self.dist_var.set("Distance Traveled: —")
        self.optim_var.set("")

    def _show_results(self):
        algo = self.last_algo
        if algo is None:
            return
        if algo.total_cost is not None:
            label = "Total Cost Used" if algo.reached else "Cost So Far (not reached)"
            self.cost_var.set(f"{label}: {algo.total_cost} moves")
            self.dist_var.set(
                f"Distance Traveled: {algo.path_length} cells  "
                f"(straight-line minimum: {algo.straight_line})")
            if algo.reached:
                if algo.total_cost <= algo.straight_line:
                    self.optim_var.set("🏆 Optimal — matches the shortest possible distance.")
                else:
                    extra = algo.total_cost - algo.straight_line
                    self.optim_var.set(
                        f"↪ {extra} move(s) longer than the straight-line minimum "
                        f"(walls forced a detour, or the algorithm is not optimal).")
            else:
                self.optim_var.set("⚠ Did not reach the goal — stats reflect the partial path.")
        else:
            self.cost_var.set("Total Cost Used: —")
            self.dist_var.set(f"❌ No path found  (straight-line distance: {algo.straight_line})")
            self.optim_var.set("")

    def _reset(self):
        self._stop()
        self.grid       = [[EMPTY] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.start      = None
        self.goal       = None
        self.overlay    = {}
        self.cost_map   = {}
        self.anim_steps = []
        self.anim_index = 0
        self.last_algo  = None
        self._draw_grid()
        self.status_var.set("Grid cleared. Place Start, Goal, and Walls.")
        self.step_var.set("Step: —")
        self._reset_results()
        self.prev_btn.config(state="disabled")
        self.next_btn.config(state="disabled")


# ──────────────────────────────────────────────
#  ALGORITHM BASE
# ──────────────────────────────────────────────
class BaseAlgo:
    def __init__(self, grid, start, goal):
        self.grid  = [row[:] for row in grid]
        self.start = start
        self.goal  = goal
        self.steps = []

        self.reached       = False
        self.total_cost    = None
        self.path_length   = None
        self.straight_line = manhattan(start, goal)

    def snap(self, overlay, cost_map, status):
        self.steps.append((dict(overlay), dict(cost_map), status))

    def solve(self):
        raise NotImplementedError

    def _mark_path(self, path, overlay):
        for cell in path:
            if cell != self.start and cell != self.goal:
                overlay[cell] = C_PATH

    def _finalize(self, path, reached):
        self.reached = reached
        if path:
            self.path_length = len(path)
            self.total_cost  = max(len(path) - 1, 0)
        else:
            self.path_length = None
            self.total_cost  = None


# ──────────────────────────────────────────────
#  1. A* SEARCH  —  f(n) = g(n) + h(n), optimal & complete
# ──────────────────────────────────────────────
class AStarSearch(BaseAlgo):
    def solve(self):
        g_score   = {self.start: 0}
        f_score   = {self.start: manhattan(self.start, self.goal)}
        came_from = {}
        closed    = set()

        counter   = 0
        open_heap = [(f_score[self.start], counter, self.start)]
        open_set  = {self.start}

        overlay  = {}
        cost_map = {}
        cost_map[self.start] = f"g=0\nf={f_score[self.start]}"

        self.snap(overlay, cost_map,
                  "A* Search started.  f(n) = g(n) + h(n)  →  g = cost-so-far, "
                  "h = Manhattan distance heuristic.")

        iteration = 0
        while open_heap:
            iteration += 1
            _, _, current = heapq.heappop(open_heap)

            if current in closed:
                continue
            open_set.discard(current)
            closed.add(current)

            if current != self.start and current != self.goal:
                overlay[current] = C_CURRENT

            g_cur = g_score[current]
            h_cur = manhattan(current, self.goal)

            if current == self.goal:
                path  = []
                node  = self.goal
                while node is not None:
                    path.append(node)
                    node = came_from.get(node)
                path.reverse()
                self._mark_path(path, overlay)
                self._finalize(path, reached=True)
                self.snap(overlay, cost_map,
                          f"🎉 Goal reached!  Path length: {len(path)} cells  |  "
                          f"Total g-cost: {g_cur}  |  Iterations: {iteration}.")
                return self.steps

            self.snap(overlay, cost_map,
                      f"Iter {iteration}: Expand {current}  |  "
                      f"g={g_cur}, h={h_cur}, f={g_cur + h_cur}  |  "
                      f"Open: {len(open_set)}, Closed: {len(closed)}")

            if current != self.start:
                overlay[current] = C_VISITED

            for nb in neighbors(self.grid, *current):
                if nb in closed:
                    continue
                tentative_g = g_cur + 1
                if tentative_g < g_score.get(nb, float("inf")):
                    came_from[nb] = current
                    g_score[nb]   = tentative_g
                    h_nb          = manhattan(nb, self.goal)
                    f_nb          = tentative_g + h_nb
                    f_score[nb]   = f_nb
                    counter      += 1
                    heapq.heappush(open_heap, (f_nb, counter, nb))
                    open_set.add(nb)
                    cost_map[nb] = f"g={tentative_g}\nf={f_nb}"
                    if nb != self.goal:
                        overlay[nb] = C_FRONTIER

            if len(self.steps) > GRID_SIZE * GRID_SIZE * 4:
                break

        self._finalize(None, reached=False)
        self.snap(overlay, cost_map,
                  "No path found — goal is unreachable from start.")
        return self.steps


# ──────────────────────────────────────────────
#  2. BREADTH-FIRST SEARCH  —  fewest steps, uniform cost
# ──────────────────────────────────────────────
class BFSSearch(BaseAlgo):
    def solve(self):
        queue     = collections.deque([self.start])
        came_from = {self.start: None}
        depth     = {self.start: 0}
        visited   = {self.start}

        overlay  = {}
        cost_map = {self.start: "d=0"}

        self.snap(overlay, cost_map,
                  "Breadth-First Search started.  Explores the shallowest "
                  "(nearest, by step-count) nodes first — guarantees the "
                  "fewest-step path when every move costs the same.")

        iteration = 0
        while queue:
            iteration += 1
            current = queue.popleft()

            if current != self.start and current != self.goal:
                overlay[current] = C_CURRENT

            d_cur = depth[current]

            if current == self.goal:
                path = []
                node = self.goal
                while node is not None:
                    path.append(node)
                    node = came_from[node]
                path.reverse()
                self._mark_path(path, overlay)
                self._finalize(path, reached=True)
                self.snap(overlay, cost_map,
                          f"🎉 Goal reached!  Path length: {len(path)} cells  |  "
                          f"Depth: {d_cur}  |  Iterations: {iteration}.")
                return self.steps

            self.snap(overlay, cost_map,
                      f"Iter {iteration}: Dequeue {current}  |  depth={d_cur}  |  "
                      f"Queue size: {len(queue)}, Visited: {len(visited)}")

            if current != self.start:
                overlay[current] = C_VISITED

            for nb in neighbors(self.grid, *current):
                if nb not in visited:
                    visited.add(nb)
                    came_from[nb] = current
                    depth[nb] = d_cur + 1
                    cost_map[nb] = f"d={d_cur + 1}"
                    queue.append(nb)
                    if nb != self.goal:
                        overlay[nb] = C_FRONTIER

            if len(self.steps) > GRID_SIZE * GRID_SIZE * 4:
                break

        self._finalize(None, reached=False)
        self.snap(overlay, cost_map,
                  "No path found — goal is unreachable from start.")
        return self.steps


# ──────────────────────────────────────────────
#  3. GREEDY BEST-FIRST SEARCH  —  uses h(n) only, NOT optimal
# ──────────────────────────────────────────────
class GreedyBestFirstSearch(BaseAlgo):
    def solve(self):
        h_start   = manhattan(self.start, self.goal)
        came_from = {}
        g_score   = {self.start: 0}
        closed    = set()

        counter   = 0
        open_heap = [(h_start, counter, self.start)]
        open_set  = {self.start}

        overlay  = {}
        cost_map = {self.start: f"h={h_start}"}

        self.snap(overlay, cost_map,
                  "Greedy Best-First Search started.  Priority = h(n) ONLY "
                  "(Manhattan distance to goal) — no regard for cost-so-far, "
                  "so the path found is fast to compute but NOT guaranteed optimal.")

        iteration = 0
        while open_heap:
            iteration += 1
            h_cur, _, current = heapq.heappop(open_heap)

            if current in closed:
                continue
            open_set.discard(current)
            closed.add(current)

            if current != self.start and current != self.goal:
                overlay[current] = C_CURRENT

            g_cur = g_score[current]

            if current == self.goal:
                path = []
                node = self.goal
                while node is not None:
                    path.append(node)
                    node = came_from.get(node)
                path.reverse()
                self._mark_path(path, overlay)
                self._finalize(path, reached=True)
                self.snap(overlay, cost_map,
                          f"🎉 Goal reached!  Path length: {len(path)} cells  |  "
                          f"(g-cost {g_cur} — not guaranteed minimal since only h was used).")
                return self.steps

            self.snap(overlay, cost_map,
                      f"Iter {iteration}: Expand {current}  |  h={h_cur}  |  "
                      f"Open: {len(open_set)}, Closed: {len(closed)}")

            if current != self.start:
                overlay[current] = C_VISITED

            for nb in neighbors(self.grid, *current):
                if nb in closed or nb in open_set:
                    continue
                came_from[nb] = current
                g_score[nb]   = g_cur + 1
                h_nb          = manhattan(nb, self.goal)
                counter      += 1
                heapq.heappush(open_heap, (h_nb, counter, nb))
                open_set.add(nb)
                cost_map[nb] = f"h={h_nb}"
                if nb != self.goal:
                    overlay[nb] = C_FRONTIER

            if len(self.steps) > GRID_SIZE * GRID_SIZE * 4:
                break

        self._finalize(None, reached=False)
        self.snap(overlay, cost_map,
                  "No path found — goal is unreachable from start.")
        return self.steps


# ──────────────────────────────────────────────
#  4. HILL CLIMBING
# ──────────────────────────────────────────────
class HillClimbing(BaseAlgo):
    def solve(self):
        current  = self.start
        visited  = set()
        path     = [current]
        overlay  = {}
        cost_map = {}

        self.snap(overlay, cost_map,
                  "Hill Climbing started. Greedy move toward goal by Manhattan distance.")

        for _ in range(GRID_SIZE * GRID_SIZE * 2):
            visited.add(current)
            nbrs = neighbors(self.grid, *current)

            for v in visited:
                if v != self.start and v != self.goal:
                    overlay[v] = C_VISITED
            for n in nbrs:
                if n not in visited and n != self.goal:
                    overlay[n] = C_FRONTIER
            overlay[current] = C_CURRENT

            for n in nbrs:
                cost_map[n] = f"h={manhattan(n, self.goal)}"
            cost_map[current] = f"h={manhattan(current, self.goal)}"

            if current == self.goal:
                self._mark_path(path, overlay)
                self._finalize(path, reached=True)
                self.snap(overlay, cost_map, "🎉 Goal reached!")
                return self.steps

            nbrs.sort(key=lambda x: manhattan(x, self.goal))
            if not nbrs:
                self.snap(overlay, cost_map,
                          "⛔ No passable neighbours. Hill Climbing stuck.")
                break

            best_nbr  = nbrs[0]
            best_dist = manhattan(best_nbr, self.goal)
            cur_dist  = manhattan(current, self.goal)

            if best_dist >= cur_dist:
                overlay[current] = C_BACKTRACK
                self.snap(overlay, cost_map,
                          f"⚠ LOCAL MINIMUM at {current}!  "
                          f"All neighbours (h={best_dist}) ≥ current (h={cur_dist}). "
                          "Cannot escape without restart.")
                break

            self.snap(overlay, cost_map,
                      f"Move {current}→{best_nbr}  |  h: {cur_dist}→{best_dist}")
            current = best_nbr
            path.append(current)

        self._mark_path(path, overlay)
        self._finalize(path, reached=False)
        return self.steps


# ──────────────────────────────────────────────
#  5. SIMULATED ANNEALING
# ──────────────────────────────────────────────
class SimulatedAnnealing(BaseAlgo):
    def solve(self):
        current  = self.start
        visited  = {current}
        path     = [current]
        overlay  = {}
        cost_map = {}

        T     = 1.0
        T_min = 0.01
        alpha = 0.92
        step  = 0

        self.snap(overlay, cost_map,
                  f"Simulated Annealing started.  T={T:.3f}.")

        while current != self.goal and T > T_min:
            nbrs = neighbors(self.grid, *current)
            if not nbrs:
                break

            for v in visited:
                if v != self.start and v != self.goal:
                    overlay[v] = C_VISITED
            for n in nbrs:
                if n != self.goal:
                    overlay[n] = C_FRONTIER
                cost_map[n] = f"h={manhattan(n, self.goal)}"
            overlay[current]  = C_CURRENT
            cost_map[current] = f"h={manhattan(current, self.goal)}"

            candidate = random.choice(nbrs)
            delta = manhattan(current, self.goal) - manhattan(candidate, self.goal)

            if delta > 0:
                accepted = True
                msg = (f"T={T:.3f} | Better move {current}→{candidate}  "
                       f"Δh={delta:+d}. Accepted.")
            else:
                prob     = math.exp(delta / T) if T > 0 else 0
                accepted = random.random() < prob
                if accepted:
                    msg = (f"T={T:.3f} | ⚡ Worse move accepted! {current}→{candidate}  "
                           f"Δh={delta:+d}, p={prob:.2f}. Escaping local min!")
                else:
                    msg = (f"T={T:.3f} | Worse move rejected. {current}→{candidate}  "
                           f"Δh={delta:+d}, p={prob:.2f}.")

            self.snap(overlay, cost_map, msg)

            if accepted:
                current = candidate
                path.append(current)
                visited.add(current)

            T    *= alpha
            step += 1

        for v in visited:
            if v != self.start and v != self.goal:
                overlay[v] = C_VISITED
        overlay[current] = C_CURRENT

        if current == self.goal:
            self._mark_path(path, overlay)
            self._finalize(path, reached=True)
            self.snap(overlay, cost_map,
                      f"🎉 Goal reached after {step} steps!  T={T:.4f}.")
        else:
            self._mark_path(path, overlay)
            self._finalize(path, reached=False)
            self.snap(overlay, cost_map,
                      f"Temperature cooled to {T:.4f}.  Ended at {current}.")
        return self.steps


# ══════════════════════════════════════════════════════════════════
#  TAB 2 — CONSTRAINT SATISFACTION PROBLEM (CSP) SOLVER
# ══════════════════════════════════════════════════════════════════
#
# Classic "map colouring" CSP (Australia regions — the standard AIMA /
# CS50 AI teaching example):
#
#   Variables   : regions (WA, NT, SA, Q, NSW, V, T)
#   Domains     : available colours (adjustable, 2-4)
#   Constraints :
#       HARD  -> neighbouring regions must be assigned different colours
#                (must never be violated — an assignment that breaks this
#                 is simply illegal)
#       SOFT  -> each region has a "preferred" colour; satisfying it is
#                desirable but NOT required — violating it costs a
#                penalty point rather than making the assignment invalid
#
# Solved with Backtracking Search + Forward Checking + the
# Minimum-Remaining-Values (MRV) heuristic, optionally guided by the
# soft preferences when ordering which colour to try first.

CSP_REGIONS = ["WA", "NT", "SA", "Q", "NSW", "V", "T"]

CSP_ADJACENCY = {
    "WA":  ["NT", "SA"],
    "NT":  ["WA", "SA", "Q"],
    "SA":  ["WA", "NT", "Q", "NSW", "V"],
    "Q":   ["NT", "SA", "NSW"],
    "NSW": ["SA", "Q", "V"],
    "V":   ["SA", "NSW"],
    "T":   [],
}

CSP_POSITIONS = {
    "WA":  (90, 230),
    "NT":  (210, 90),
    "SA":  (240, 240),
    "Q":   (350, 100),
    "NSW": (380, 240),
    "V":   (330, 340),
    "T":   (380, 410),
}

# Fixed default colour PREFERENCES (index into the active colour list,
# wrapped by modulo so it still works for smaller domain sizes)
CSP_PREFERENCES_IDX = {
    "WA": 0, "NT": 1, "SA": 2, "Q": 0, "NSW": 1, "V": 2, "T": 3,
}


class CSPSolver:
    """Backtracking Search + Forward Checking + MRV for map colouring."""

    def __init__(self, domain_colors, use_soft):
        self.regions   = CSP_REGIONS
        self.adjacency = CSP_ADJACENCY
        self.domain_colors = domain_colors          # list of hex colours
        self.use_soft  = use_soft
        self.preferences = {
            r: domain_colors[CSP_PREFERENCES_IDX[r] % len(domain_colors)]
            for r in self.regions
        }

        self.steps = []            # list of (assignment_copy, current_var, event, message)
        self.backtrack_count = 0
        self.attempt_count   = 0
        self.success = False
        self.final_assignment = None

    def snap(self, assignment, current_var, event, message):
        self.steps.append((dict(assignment), current_var, event, message))

    def is_consistent(self, region, color, assignment):
        for nb in self.adjacency[region]:
            if assignment.get(nb) == color:
                return False
        return True

    def forward_check(self, region, color, domains, assignment):
        """Remove `color` from unassigned neighbours' domains. Returns removals."""
        removed = {}
        for nb in self.adjacency[region]:
            if nb not in assignment and color in domains[nb]:
                domains[nb].remove(color)
                removed.setdefault(nb, []).append(color)
        return removed

    def restore(self, domains, removed):
        for region, colors in removed.items():
            domains[region].extend(colors)

    def select_variable(self, assignment, domains):
        unassigned = [r for r in self.regions if r not in assignment]
        # MRV: fewest legal values left; tie-break by degree (more neighbours first)
        return min(unassigned,
                   key=lambda r: (len(domains[r]), -len(self.adjacency[r])))

    def order_values(self, region, domains):
        vals = list(domains[region])
        if self.use_soft:
            pref = self.preferences.get(region)
            vals.sort(key=lambda c: 0 if c == pref else 1)
        return vals

    def backtrack(self, assignment, domains):
        if len(assignment) == len(self.regions):
            self.success = True
            self.final_assignment = dict(assignment)
            return True

        var = self.select_variable(assignment, domains)
        self.snap(assignment, var, "select",
                  f"เลือกตัวแปร (MRV) : {var}  |  ค่าที่เหลือในโดเมน: {len(domains[var])}")

        for value in self.order_values(var, domains):
            self.attempt_count += 1
            pref_note = ""
            if self.use_soft:
                pref_note = ("  (ตรงกับสีที่ต้องการ ✓)"
                             if value == self.preferences.get(var) else
                             "  (ไม่ตรงสีที่ต้องการ — soft penalty)")

            if not self.is_consistent(var, value, assignment):
                self.snap(assignment, var, "conflict",
                          f"ลอง {var} = {value}  →  ✗ ขัดกับ Hard Constraint "
                          f"(พื้นที่ข้างเคียงใช้สีนี้แล้ว). ข้ามค่านี้.")
                continue

            assignment[var] = value
            self.snap(assignment, var, "assign",
                      f"กำหนดค่า {var} = {value}{pref_note}")

            removed = self.forward_check(var, value, domains, assignment)
            wiped = [nb for nb in removed if len(domains[nb]) == 0]

            if wiped:
                self.snap(assignment, var, "prune_fail",
                          f"Forward Checking: {var}={value} ทำให้โดเมนของ "
                          f"{', '.join(wiped)} ว่างเปล่า → ย้อนกลับทันที (backtrack).")
                self.restore(domains, removed)
                del assignment[var]
                self.backtrack_count += 1
                continue

            if removed:
                pruned_desc = ", ".join(f"{nb}(-{len(c)})" for nb, c in removed.items())
                self.snap(assignment, var, "prune_ok",
                          f"Forward Checking: ตัดค่าที่เป็นไปไม่ได้ออกจากเพื่อนบ้าน: {pruned_desc}")

            if self.backtrack(assignment, domains):
                return True

            self.restore(domains, removed)
            del assignment[var]
            self.backtrack_count += 1
            self.snap(assignment, var, "backtrack",
                      f"ย้อนกลับ (backtrack) จาก {var} = {value}  |  "
                      f"จำนวนครั้งที่ backtrack: {self.backtrack_count}")

        return False

    def solve(self):
        domains = {r: list(self.domain_colors) for r in self.regions}
        assignment = {}
        self.snap(assignment, None, "start",
                  f"เริ่ม Backtracking Search + Forward Checking + MRV  |  "
                  f"จำนวนสี (โดเมน): {len(self.domain_colors)}  |  "
                  f"Soft Constraints: {'เปิดใช้งาน' if self.use_soft else 'ปิด'}")
        ok = self.backtrack(assignment, domains)
        if ok:
            soft_satisfied = sum(
                1 for r in self.regions
                if self.final_assignment.get(r) == self.preferences.get(r)
            )
            self.snap(self.final_assignment, None, "success",
                      f"🎉 พบคำตอบที่สอดคล้องกับ Hard Constraints ทั้งหมด!  "
                      f"Soft preference ที่พึงพอใจ: {soft_satisfied}/{len(self.regions)}  |  "
                      f"Backtrack ทั้งหมด: {self.backtrack_count}")
        else:
            self.snap(assignment, None, "fail",
                      f"❌ ไม่พบคำตอบที่สอดคล้องกับ Hard Constraints ทั้งหมด "
                      f"(โดเมนสีอาจเล็กเกินไป)  |  Backtrack ทั้งหมด: {self.backtrack_count}")
        return self.steps


class CSPApp:
    """Everything for TAB 2 — the adjustable CSP visualizer."""

    def __init__(self, parent):
        self.parent = parent

        self.running    = False
        self.paused     = False
        self.after_id   = None
        self.steps      = []
        self.step_index = 0
        self.solver     = None

        self._build_ui()
        self._draw_graph({})

    # ── UI ────────────────────────────────────────────────
    def _build_ui(self):
        canvas_frame = tk.Frame(self.parent, bg=C_BG)
        canvas_frame.grid(row=0, column=0, padx=(16, 8), pady=16, sticky="n")

        self.canvas_w, self.canvas_h = 460, 460
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_w, height=self.canvas_h,
                                bg=C_GRID_LINE, highlightthickness=0)
        self.canvas.pack()

        ctrl = tk.Frame(self.parent, bg=C_PANEL, width=320)
        ctrl.grid(row=0, column=1, padx=(8, 16), pady=16, sticky="ns")
        ctrl.grid_propagate(False)

        pad = dict(padx=12, pady=4)

        tk.Label(ctrl, text="CS50 AI", bg=C_PANEL, fg=C_ACCENT,
                 font=("Helvetica", 11, "bold")).pack(**pad, anchor="w")
        tk.Label(ctrl, text="CSP Map-Colouring Solver", bg=C_PANEL, fg=C_TEXT_MAIN,
                 font=("Helvetica", 14, "bold")).pack(padx=12, pady=(0, 2), anchor="w")
        tk.Label(ctrl,
                 text="ปัญหา: กำหนดสีให้แต่ละพื้นที่บนแผนที่ โดยพื้นที่ที่ติดกัน "
                      "ต้องได้สีต่างกัน (Hard Constraint) และควรตรงกับสีที่ต้องการ "
                      "หากเปิดใช้ Soft Constraint",
                 bg=C_PANEL, fg=C_TEXT_DIM, font=("Helvetica", 8),
                 wraplength=280, justify="left").pack(padx=12, pady=(0, 4), anchor="w")
        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=5)

        # ── Domain size ──
        tk.Label(ctrl, text="DOMAIN SIZE (จำนวนสี)", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9, "bold")).pack(**pad, anchor="w")
        self.domain_size_var = tk.IntVar(value=3)
        dsize_frame = tk.Frame(ctrl, bg=C_PANEL)
        dsize_frame.pack(padx=12, pady=2, fill="x")
        for n in (2, 3, 4):
            tk.Radiobutton(dsize_frame, text=f"{n} สี", variable=self.domain_size_var,
                           value=n, bg=C_PANEL, fg=C_TEXT_MAIN, selectcolor=C_ACCENT,
                           activebackground=C_PANEL, activeforeground=C_TEXT_MAIN,
                           font=("Helvetica", 10), cursor="hand2",
                           command=self._preview_domain
                           ).pack(side="left", padx=(0, 10))

        # colour preview swatches
        self.swatch_frame = tk.Frame(ctrl, bg=C_PANEL)
        self.swatch_frame.pack(padx=12, pady=(0, 4), fill="x")
        self._preview_domain()

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=5)

        # ── Constraint set ──
        tk.Label(ctrl, text="CONSTRAINT SET", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9, "bold")).pack(**pad, anchor="w")
        tk.Label(ctrl, text="✓ Hard: พื้นที่ติดกันห้ามใช้สีเดียวกัน (ตายตัว)",
                 bg=C_PANEL, fg=C_TEXT_MAIN, font=("Helvetica", 9),
                 wraplength=280, justify="left").pack(padx=12, anchor="w")
        self.soft_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl, text="เปิดใช้ Soft Constraint (สีที่ต้องการต่อพื้นที่)",
                       variable=self.soft_var, bg=C_PANEL, fg=C_TEXT_MAIN,
                       selectcolor=C_ACCENT, activebackground=C_PANEL,
                       activeforeground=C_TEXT_MAIN, font=("Helvetica", 9),
                       wraplength=270, justify="left", cursor="hand2"
                       ).pack(padx=12, pady=(2, 0), anchor="w")

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=5)

        # ── Speed ──
        tk.Label(ctrl, text="ANIMATION SPEED", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9, "bold")).pack(**pad, anchor="w")
        spd_frame = tk.Frame(ctrl, bg=C_PANEL)
        spd_frame.pack(padx=12, pady=2, fill="x")
        tk.Label(spd_frame, text="Fast", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 8)).pack(side="left")
        self.speed_var = tk.IntVar(value=300)
        tk.Scale(spd_frame, variable=self.speed_var, from_=20, to=1200,
                 orient="horizontal", bg=C_PANEL, fg=C_TEXT_MAIN,
                 troughcolor=C_ACCENT, highlightthickness=0,
                 showvalue=False).pack(side="left", fill="x", expand=True)
        tk.Label(spd_frame, text="Slow", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 8)).pack(side="left")

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=5)

        # ── Primary buttons ──
        btn_cfg = dict(font=("Helvetica", 11, "bold"), bd=0, cursor="hand2",
                       pady=7, relief="flat")
        self.run_btn = tk.Button(ctrl, text="▶  SOLVE",
                                 bg="#00b894", fg="white",
                                 command=self._run, **btn_cfg)
        self.run_btn.pack(padx=12, pady=3, fill="x")

        self.pause_btn = tk.Button(ctrl, text="⏸  PAUSE",
                                   bg="#6c5ce7", fg="white",
                                   command=self._toggle_pause,
                                   state="disabled", **btn_cfg)
        self.pause_btn.pack(padx=12, pady=3, fill="x")

        self.stop_btn = tk.Button(ctrl, text="■  STOP",
                                  bg="#d63031", fg="white",
                                  command=self._stop,
                                  state="disabled", **btn_cfg)
        self.stop_btn.pack(padx=12, pady=3, fill="x")

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=5)
        nav_frame = tk.Frame(ctrl, bg=C_PANEL)
        nav_frame.pack(padx=12, pady=3, fill="x")
        nav_btn = dict(font=("Helvetica", 10, "bold"), bd=0, cursor="hand2",
                       pady=6, relief="flat", bg=C_ACCENT, fg=C_TEXT_MAIN)
        self.prev_btn = tk.Button(nav_frame, text="◀  PREV",
                                  command=self._step_back,
                                  state="disabled", **nav_btn)
        self.prev_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.next_btn = tk.Button(nav_frame, text="NEXT  ▶",
                                  command=self._step_forward,
                                  state="disabled", **nav_btn)
        self.next_btn.pack(side="left", expand=True, fill="x")

        tk.Button(ctrl, text="↺  RESET", bg=C_ACCENT, fg=C_TEXT_MAIN,
                  command=self._reset, **btn_cfg).pack(padx=12, pady=(8, 3), fill="x")

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=5)

        tk.Label(ctrl, text="STATUS", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9, "bold")).pack(**pad, anchor="w")
        self.status_var = tk.StringVar(value="พร้อมแล้ว. เลือกจำนวนสีและกด SOLVE.")
        tk.Label(ctrl, textvariable=self.status_var, bg=C_ACCENT, fg=C_TEXT_MAIN,
                 font=("Helvetica", 9), wraplength=280, justify="left", anchor="nw"
                 ).pack(padx=12, pady=4, fill="x", ipadx=8, ipady=8)

        self.step_var = tk.StringVar(value="Step: —")
        tk.Label(ctrl, textvariable=self.step_var, bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9)).pack(padx=12, anchor="w")

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=5)
        tk.Label(ctrl, text="RESULTS", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9, "bold")).pack(**pad, anchor="w")
        results_box = tk.Frame(ctrl, bg=C_ACCENT)
        results_box.pack(padx=12, pady=4, fill="x")
        self.result_var = tk.StringVar(value="—")
        tk.Label(results_box, textvariable=self.result_var, bg=C_ACCENT, fg=C_TEXT_MAIN,
                 font=("Helvetica", 9), anchor="w", justify="left", wraplength=280
                 ).pack(fill="x", padx=8, pady=8)

    def _preview_domain(self):
        for w in self.swatch_frame.winfo_children():
            w.destroy()
        n = self.domain_size_var.get()
        for hexcolor, name in CSP_COLOR_PALETTE[:n]:
            f = tk.Frame(self.swatch_frame, bg=C_PANEL)
            f.pack(side="left", padx=(0, 8))
            tk.Label(f, bg=hexcolor, width=2, height=1).pack(side="left", padx=(0, 3))
            tk.Label(f, text=name, bg=C_PANEL, fg=C_TEXT_MAIN,
                     font=("Helvetica", 8)).pack(side="left")

    # ── DRAWING ──────────────────────────────────────────
    def _draw_graph(self, assignment, current_var=None, event=None):
        self.canvas.delete("all")

        # edges
        drawn = set()
        for region, nbs in CSP_ADJACENCY.items():
            for nb in nbs:
                key = tuple(sorted((region, nb)))
                if key in drawn:
                    continue
                drawn.add(key)
                x1, y1 = CSP_POSITIONS[region]
                x2, y2 = CSP_POSITIONS[nb]
                conflict = (assignment.get(region) is not None and
                            assignment.get(region) == assignment.get(nb))
                color = C_CSP_EDGE_CONFLICT if conflict else C_CSP_EDGE
                width = 3 if conflict else 2
                self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

        # nodes
        radius = 32
        for region, (x, y) in CSP_POSITIONS.items():
            fill = assignment.get(region, C_CSP_NODE_DEFAULT)
            border = C_CSP_NODE_CURRENT if region == current_var else C_CSP_NODE_BORDER
            bw = 4 if region == current_var else 2
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                    fill=fill, outline=border, width=bw)
            text_fill = C_BG if region in assignment else C_TEXT_MAIN
            self.canvas.create_text(x, y, text=region, fill=text_fill,
                                    font=("Helvetica", 13, "bold"))

    # ── CONTROLS ─────────────────────────────────────────
    def _run(self):
        self._stop()
        n = self.domain_size_var.get()
        colors = [c for c, _ in CSP_COLOR_PALETTE[:n]]
        use_soft = self.soft_var.get()

        self.solver  = CSPSolver(colors, use_soft)
        self.steps   = self.solver.solve()
        self.step_index = 0
        self.result_var.set("—")

        self.running = True
        self.paused  = False
        self.run_btn.config(state="disabled")
        self.pause_btn.config(state="normal", text="⏸  PAUSE", bg="#6c5ce7")
        self.stop_btn.config(state="normal")
        self.prev_btn.config(state="disabled")
        self.next_btn.config(state="disabled")

        self._animate()

    def _animate(self):
        if not self.running or self.paused:
            return
        if self.step_index >= len(self.steps):
            self._finish()
            return
        self._apply_step(self.step_index)
        self.step_index += 1
        self.after_id = self.parent.after(self.speed_var.get(), self._animate)

    def _apply_step(self, idx):
        assignment, current_var, event, message = self.steps[idx]
        self._draw_graph(assignment, current_var, event)
        self.status_var.set(message)
        self.step_var.set(f"Step: {idx + 1} / {len(self.steps)}")

    def _finish(self):
        self.running = False
        self.paused  = False
        self.run_btn.config(state="normal")
        self.pause_btn.config(state="disabled", text="⏸  PAUSE")
        self.stop_btn.config(state="disabled")
        self._update_nav_buttons()
        self._show_results()

    def _show_results(self):
        s = self.solver
        if s is None:
            return
        if s.success:
            soft_satisfied = sum(
                1 for r in CSP_REGIONS
                if s.final_assignment.get(r) == s.preferences.get(r)
            )
            txt = (f"✅ พบคำตอบที่ถูกต้อง (Hard Constraints ผ่านทั้งหมด)\n"
                   f"จำนวนครั้งที่ Backtrack: {s.backtrack_count}\n"
                   f"จำนวนครั้งที่ลองกำหนดค่า: {s.attempt_count}")
            if s.use_soft:
                txt += f"\nSoft preference พึงพอใจ: {soft_satisfied}/{len(CSP_REGIONS)}"
            self.result_var.set(txt)
        else:
            self.result_var.set(
                f"❌ ไม่พบคำตอบที่สอดคล้อง Hard Constraints ทั้งหมด\n"
                f"จำนวนครั้งที่ Backtrack: {s.backtrack_count}\n"
                f"ลองเพิ่มจำนวนสี (โดเมน) แล้วลองใหม่")

    def _toggle_pause(self):
        if not self.running:
            return
        self.paused = not self.paused
        if self.paused:
            if self.after_id:
                self.parent.after_cancel(self.after_id)
                self.after_id = None
            self.pause_btn.config(text="▶  RESUME", bg="#00b894")
            self._update_nav_buttons()
            self.status_var.set("⏸  Paused. ใช้ ◀ PREV / NEXT ▶ เพื่อดูทีละขั้น.")
        else:
            self.pause_btn.config(text="⏸  PAUSE", bg="#6c5ce7")
            self.prev_btn.config(state="disabled")
            self.next_btn.config(state="disabled")
            self._animate()

    def _stop(self):
        was_running = self.running
        if self.after_id:
            self.parent.after_cancel(self.after_id)
            self.after_id = None
        self.running = False
        self.paused  = False
        self.run_btn.config(state="normal")
        self.pause_btn.config(state="disabled", text="⏸  PAUSE", bg="#6c5ce7")
        self.stop_btn.config(state="disabled")
        self._update_nav_buttons()
        if was_running:
            self.status_var.set("⏹  Stopped.")
            self._show_results()

    def _step_back(self):
        if self.step_index > 1:
            self.step_index -= 1
            self._apply_step(self.step_index - 1)
        self._update_nav_buttons()

    def _step_forward(self):
        if self.step_index < len(self.steps):
            self._apply_step(self.step_index)
            self.step_index += 1
        self._update_nav_buttons()

    def _update_nav_buttons(self):
        can_nav = (not self.running or self.paused) and len(self.steps) > 0
        self.prev_btn.config(state="normal" if (can_nav and self.step_index > 1) else "disabled")
        self.next_btn.config(state="normal" if (can_nav and self.step_index < len(self.steps)) else "disabled")

    def _reset(self):
        self._stop()
        self.steps = []
        self.step_index = 0
        self.solver = None
        self._draw_graph({})
        self.status_var.set("รีเซ็ตแล้ว. เลือกจำนวนสีและกด SOLVE.")
        self.step_var.set("Step: —")
        self.result_var.set("—")
        self.prev_btn.config(state="disabled")
        self.next_btn.config(state="disabled")


# ──────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    root.title("CS50 AI — Optimization & Constraint Satisfaction Visualizer")
    root.configure(bg=C_BG)

    # ── Fit the window to the current screen ──────────────────────
    root.resizable(True, True)
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()

    # Shrink the grid cell size on smaller screens/laptops so the whole
    # board tends to fit without scrolling. On very small screens the
    # ScrollableFrame below still guarantees nothing gets cut off.
    reserved_h = 210   # tabs bar + padding + window chrome allowance
    reserved_w = 430   # control panel width + paddings
    avail_h = max(320, screen_h - reserved_h)
    avail_w = max(320, screen_w - reserved_w)
    CELL_PX = max(20, min(42, avail_h // GRID_SIZE, avail_w // GRID_SIZE))

    win_w = min(screen_w - 60, 1320)
    win_h = min(screen_h - 100, 900)
    pos_x = max(0, (screen_w - win_w) // 2)
    pos_y = max(0, (screen_h - win_h) // 2)
    root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
    root.minsize(640, 420)

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TCombobox",
                    fieldbackground=C_ACCENT,
                    background=C_ACCENT,
                    foreground=C_TEXT_MAIN,
                    selectbackground=C_ACCENT,
                    selectforeground=C_TEXT_MAIN)
    style.configure("TSeparator", background=C_ACCENT)
    style.configure("TNotebook", background=C_BG, borderwidth=0)
    style.configure("TNotebook.Tab", background=C_PANEL, foreground=C_TEXT_MAIN,
                    padding=[16, 8], font=("Helvetica", 10, "bold"))
    style.map("TNotebook.Tab",
             background=[("selected", C_ACCENT)],
             foreground=[("selected", C_TEXT_MAIN)])

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    tab1 = tk.Frame(notebook, bg=C_BG)
    tab2 = tk.Frame(notebook, bg=C_BG)
    notebook.add(tab1, text="🧭  Pathfinding / Optimization")
    notebook.add(tab2, text="🗺️  CSP Solver (Adjustable)")

    # Each tab's real content lives inside a ScrollableFrame, so if the
    # window is ever smaller than the content (small laptop, half-screen
    # snap, etc.) the user can scroll instead of losing buttons/results.
    scroll1 = ScrollableFrame(tab1, bg=C_BG)
    scroll1.pack(fill="both", expand=True)
    scroll2 = ScrollableFrame(tab2, bg=C_BG)
    scroll2.pack(fill="both", expand=True)

    pathfinding_app = PathfindingApp(scroll1.body)
    csp_app         = CSPApp(scroll2.body)

    root.mainloop()
