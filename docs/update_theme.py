import re
import os

filepath = "emergi_docs.html"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

new_css = """
        * { box-sizing: border-box; margin: 0; padding: 0; }
        :root {
            /* Sleek Obsidian / Slate theme */
            --bg: #030712;
            --surface: rgba(17, 24, 39, 0.6);
            --surface2: rgba(31, 41, 55, 0.7);
            --surface3: rgba(55, 65, 81, 0.8);
            --border: rgba(255, 255, 255, 0.08);
            --border2: rgba(255, 255, 255, 0.15);
            --text: #f9fafb;
            --muted: #9ca3af;
            --hint: #6b7280;
            
            /* Vibrant Accents */
            --red: #f87171;
            --red-bg: rgba(248, 113, 113, 0.15);
            --amber: #fbbf24;
            --amber-bg: rgba(251, 191, 36, 0.15);
            --green: #34d399;
            --green-bg: rgba(52, 211, 153, 0.15);
            --blue: #38bdf8;
            --blue-bg: rgba(56, 189, 248, 0.15);
            --purple: #c084fc;
            --purple-bg: rgba(192, 132, 252, 0.15);
            --teal: #2dd4bf;
            --teal-bg: rgba(45, 212, 191, 0.15);
            
            --radius: 12px;
            --radius-lg: 20px;
            --font: 'Outfit', 'Inter', system-ui, sans-serif;
            --mono: 'JetBrains Mono', 'Fira Code', monospace;
            --glow: 0 0 20px rgba(56, 189, 248, 0.3);
            --glow-red: 0 0 20px rgba(248, 113, 113, 0.2);
        }

        body {
            background-color: var(--bg);
            background-image: 
                radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.08) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(192, 132, 252, 0.08) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(248, 113, 113, 0.05) 0px, transparent 50%),
                linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
            background-size: 100% 100%, 100% 100%, 100% 100%, 30px 30px, 30px 30px;
            background-attachment: fixed;
            color: var(--text);
            font-family: var(--font);
            font-size: 15px;
            line-height: 1.6;
            min-height: 100vh;
        }

        /* ── SCROLLBAR ─────────────────────────── */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

        /* ── TOP NAV ───────────────────────────── */
        nav {
            position: sticky;
            top: 0;
            z-index: 100;
            background: rgba(3, 7, 18, 0.6);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--border);
            padding: 0 32px;
            display: flex;
            align-items: center;
            gap: 20px;
            height: 64px;
        }
        .nav-logo {
            display: flex;
            align-items: center;
            gap: 12px;
            text-decoration: none;
        }
        .nav-logo .pulse-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--red);
            box-shadow: 0 0 15px var(--red);
            animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
            flex-shrink: 0;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; box-shadow: 0 0 15px rgba(248, 113, 113, 0.6); transform: scale(1); }
            50% { opacity: .6; box-shadow: 0 0 30px rgba(248, 113, 113, 0); transform: scale(1.1); }
        }
        .nav-logo span {
            font-size: 18px;
            font-weight: 700;
            letter-spacing: 0.5px;
            color: #fff;
            background: linear-gradient(to right, #fff, #9ca3af);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .nav-logo small {
            font-size: 12px;
            font-family: var(--mono);
            color: var(--blue);
            background: var(--blue-bg);
            padding: 2px 6px;
            border-radius: 6px;
            font-weight: 500;
            margin-left: 8px;
        }
        .nav-links {
            display: flex;
            gap: 4px;
            margin-left: auto;
        }
        .nav-links a {
            color: var(--muted);
            text-decoration: none;
            font-size: 14px;
            font-weight: 500;
            padding: 8px 16px;
            border-radius: 8px;
            transition: all 0.2s ease;
            position: relative;
        }
        .nav-links a:hover {
            color: var(--text);
            background: rgba(255,255,255,0.05);
        }
        .nav-links a.active {
            color: var(--blue);
            background: var(--blue-bg);
            box-shadow: inset 0 0 0 1px rgba(56, 189, 248, 0.2);
        }

        #live-status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            font-weight: 600;
            color: var(--muted);
            padding: 6px 16px;
            border-radius: 20px;
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--border);
            margin-left: 16px;
            transition: all 0.3s ease;
        }
        #live-status .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--hint);
        }
        #live-status.online {
            border-color: rgba(52, 211, 153, 0.3);
            background: rgba(52, 211, 153, 0.05);
            color: var(--green);
        }
        #live-status.online .dot {
            background: var(--green);
            box-shadow: 0 0 10px var(--green);
            animation: pulse-green 2s infinite;
        }
        @keyframes pulse-green {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: .5; transform: scale(0.8); }
        }

        /* ── LAYOUT ─────────────────────────────── */
        .page { max-width: 1400px; margin: 0 auto; padding: 48px 32px; animation: fadeUp 0.6s ease-out; }
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .section { margin-bottom: 64px; }
        .section-header {
            display: flex;
            align-items: baseline;
            gap: 16px;
            margin-bottom: 24px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
        }
        .section-header h2 {
            font-size: 24px;
            font-weight: 700;
            color: #fff;
            letter-spacing: -0.5px;
        }
        .section-header p { font-size: 14px; color: var(--muted); }

        /* ── HERO ───────────────────────────────── */
        .hero {
            background: var(--surface);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border);
            border-top: 1px solid rgba(255,255,255,0.15);
            border-radius: 24px;
            padding: 56px 48px;
            margin-bottom: 48px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        }
        .hero::before {
            content: '';
            position: absolute;
            top: -100px;
            right: -100px;
            width: 500px;
            height: 500px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(56, 189, 248, 0.15) 0%, transparent 70%);
            pointer-events: none;
        }
        .hero-tag {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            color: var(--blue);
            background: var(--blue-bg);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 24px;
            padding: 6px 16px;
            margin-bottom: 24px;
            box-shadow: 0 0 20px rgba(56, 189, 248, 0.1);
        }
        .hero h1 {
            font-size: 42px;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 16px;
            letter-spacing: -1px;
            color: #fff;
        }
        .hero h1 span {
            background: linear-gradient(135deg, var(--red), var(--purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero .subtitle {
            font-size: 16px;
            color: var(--muted);
            max-width: 700px;
            line-height: 1.8;
            margin-bottom: 40px;
        }
        .hero-stats { display: flex; gap: 48px; flex-wrap: wrap; }
        .hero-stat { display: flex; flex-direction: column; gap: 4px; }
        .hero-stat .val {
            font-size: 32px;
            font-weight: 800;
            color: #fff;
            font-family: var(--mono);
            background: linear-gradient(180deg, #fff, #9ca3af);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-stat .lbl {
            font-size: 12px;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }

        /* ── GRID HELPERS ───────────────────────── */
        .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; }
        .grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 24px; }
        .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; }
        @media(max-width:1024px) {
            .grid-3 { grid-template-columns: repeat(2, 1fr); }
            .grid-4 { grid-template-columns: repeat(2, 1fr); }
        }
        @media(max-width:768px) {
            .grid-3, .grid-4, .grid-2 { grid-template-columns: 1fr; }
            .hero { padding: 32px 24px; }
            .hero h1 { font-size: 32px; }
        }

        /* ── CARDS ──────────────────────────────── */
        .card, .metric-card, .task-card, .feat-card {
            background: var(--surface);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--border);
            border-top: 1px solid rgba(255,255,255,0.1);
            border-radius: var(--radius-lg);
            padding: 24px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        .card:hover, .task-card:hover, .feat-card:hover {
            transform: translateY(-4px);
            border-color: rgba(255,255,255,0.2);
            box-shadow: 0 10px 30px rgba(0,0,0,0.5), 0 0 20px rgba(255,255,255,0.03);
        }
        
        /* Metric Card specific styling */
        .metric-card {
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .metric-card .val {
            font-size: 36px;
            font-weight: 800;
            font-family: var(--mono);
            color: #fff;
            text-shadow: 0 0 15px rgba(255,255,255,0.2);
        }
        .metric-card .lbl { font-size: 13px; color: var(--muted); margin-top: 4px; font-weight: 500; }
        .metric-card .delta {
            font-size: 12px; margin-top: 8px; font-weight: 600; padding: 4px 10px;
            border-radius: 12px; display: inline-flex; width: fit-content;
        }
        .delta.up { background: var(--green-bg); color: var(--green); }
        .delta.down { background: var(--red-bg); color: var(--red); }
        .delta.neutral { background: rgba(255,255,255,0.05); color: var(--muted); }

        /* ── TASK CARDS ─────────────────────────── */
        .task-card {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .task-card::before {
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 4px;
            transition: all 0.3s;
        }
        .task-card.easy::before { background: var(--green); box-shadow: 0 0 10px var(--green); }
        .task-card.medium::before { background: var(--amber); box-shadow: 0 0 10px var(--amber); }
        .task-card.hard::before { background: var(--red); box-shadow: 0 0 10px var(--red); }

        .task-header { display: flex; align-items: flex-start; gap: 12px; }
        .task-num {
            min-width: 36px; height: 36px; border-radius: 10px; display: flex; align-items: center;
            justify-content: center; font-size: 14px; font-weight: 800; font-family: var(--mono);
        }
        .easy .task-num { background: var(--green-bg); color: var(--green); }
        .medium .task-num { background: var(--amber-bg); color: var(--amber); }
        .hard .task-num { background: var(--red-bg); color: var(--red); }
        
        .task-title { font-size: 16px; font-weight: 700; color: #fff; line-height: 1.3; margin-bottom: 4px; }
        .task-desc { font-size: 14px; color: var(--muted); line-height: 1.6; flex-grow: 1; }
        
        .task-footer { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-top: auto; }
        
        .chip {
            font-size: 11px; padding: 4px 10px; border-radius: 6px; font-weight: 600;
            letter-spacing: 0.5px; text-transform: uppercase; border: 1px solid transparent;
            font-family: var(--mono); transition: all 0.2s;
        }
        .chip:hover { filter: brightness(1.2); }
        .chip-green { background: var(--green-bg); color: var(--green); border-color: rgba(52, 211, 153, 0.3); box-shadow: 0 0 10px rgba(52, 211, 153, 0.1); }
        .chip-amber { background: var(--amber-bg); color: var(--amber); border-color: rgba(251, 191, 36, 0.3); box-shadow: 0 0 10px rgba(251, 191, 36, 0.1); }
        .chip-red { background: var(--red-bg); color: var(--red); border-color: rgba(248, 113, 113, 0.3); box-shadow: 0 0 10px rgba(248, 113, 113, 0.1); }
        .chip-blue { background: var(--blue-bg); color: var(--blue); border-color: rgba(56, 189, 248, 0.3); box-shadow: 0 0 10px rgba(56, 189, 248, 0.1); }
        .chip-purple { background: var(--purple-bg); color: var(--purple); border-color: rgba(192, 132, 252, 0.3); box-shadow: 0 0 10px rgba(192, 132, 252, 0.1); }
        .chip-teal { background: var(--teal-bg); color: var(--teal); border-color: rgba(45, 212, 191, 0.3); box-shadow: 0 0 10px rgba(45, 212, 191, 0.1); }
        .chip-gray { background: rgba(255,255,255,0.05); color: var(--muted); border-color: var(--border); }

        .baseline-bar { display: flex; align-items: center; gap: 12px; margin-top: 8px; background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 8px; border: 1px solid var(--border); }
        .baseline-bar span { font-size: 11px; font-weight: 600; text-transform: uppercase; color: var(--hint); }
        .baseline-bar .bar-track { flex: 1; height: 6px; background: rgba(255,255,255,0.05); border-radius: 3px; overflow: hidden; }
        .baseline-bar .bar-fill { height: 100%; border-radius: 3px; transition: width 1s cubic-bezier(0.4, 0, 0.2, 1); position: relative; }
        .easy .bar-fill { background: linear-gradient(90deg, #059669, #34d399); box-shadow: 0 0 10px var(--green); }
        .medium .bar-fill { background: linear-gradient(90deg, #d97706, #fbbf24); box-shadow: 0 0 10px var(--amber); }
        .hard .bar-fill { background: linear-gradient(90deg, #dc2626, #f87171); box-shadow: 0 0 10px var(--red); }
        .baseline-bar .bar-val { font-size: 12px; color: #fff; font-family: var(--mono); min-width: 38px; text-align: right; }

        /* ── API CONSOLE ────────────────────────── */
        .console {
            background: #020617; /* Very dark background to mimic terminal */
            border: 1px solid var(--border);
            border-top: 1px solid rgba(255,255,255,0.15);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
            position: relative;
        }
        /* Fake macOS window controls */
        .console::before {
            content: '';
            display: block;
            height: 36px;
            background: rgba(30, 41, 59, 1);
            border-bottom: 1px solid var(--border);
        }
        .console::after {
            content: '';
            position: absolute;
            top: 12px; left: 16px;
            width: 12px; height: 12px;
            border-radius: 50%;
            background: #ef4444;
            box-shadow: 20px 0 0 #f59e0b, 40px 0 0 #10b981;
        }

        .console-tabs {
            display: flex;
            background: rgba(15, 23, 42, 0.9);
            padding: 0 12px;
            border-bottom: 1px solid var(--border);
            overflow-x: auto;
        }
        .console-tab {
            padding: 14px 24px;
            font-size: 13px;
            font-family: var(--mono);
            font-weight: 500;
            color: var(--hint);
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
            user-select: none;
            white-space: nowrap;
        }
        .console-tab:hover { color: var(--text); background: rgba(255,255,255,0.03); }
        .console-tab.active { color: var(--blue); border-bottom-color: var(--blue); text-shadow: 0 0 10px rgba(56, 189, 248, 0.4); }

        .console-body { padding: 32px; display: none; animation: fadeIn 0.3s ease; }
        .console-body.active { display: block; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

        .console-row {
            display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; align-items: flex-end;
            background: rgba(255,255,255,0.02); padding: 16px; border-radius: 12px; border: 1px dashed var(--border);
        }
        .console-row label {
            font-size: 13px; font-weight: 600; color: var(--text); display: block; margin-bottom: 8px;
            font-family: var(--font);
        }
        .console-row select, .console-row input {
            background: rgba(0,0,0,0.5);
            border: 1px solid rgba(255,255,255,0.1);
            color: var(--text);
            border-radius: 8px;
            padding: 10px 16px;
            font-size: 14px;
            font-family: var(--mono);
            outline: none;
            transition: all 0.2s;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
        }
        .console-row select:focus, .console-row input:focus {
            border-color: var(--blue);
            box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2), inset 0 2px 4px rgba(0,0,0,0.2);
        }

        .btn {
            display: inline-flex; align-items: center; gap: 8px; padding: 10px 24px;
            border-radius: 8px; font-size: 14px; font-weight: 600; border: none; cursor: pointer;
            transition: all 0.2s; font-family: var(--font); text-transform: uppercase; letter-spacing: 0.5px;
            position: relative; overflow: hidden;
        }
        .btn::after {
            content: ''; position: absolute; inset: 0;
            background: linear-gradient(rgba(255,255,255,0.2), transparent); opacity: 0; transition: 0.2s;
        }
        .btn:hover::after { opacity: 1; }
        .btn:active { transform: scale(0.97); }

        .btn-primary { background: linear-gradient(135deg, var(--blue), #2563eb); color: #fff; box-shadow: 0 0 20px rgba(56, 189, 248, 0.4); }
        .btn-danger { background: linear-gradient(135deg, var(--red), #991b1b); color: #fff; box-shadow: 0 0 20px rgba(248, 113, 113, 0.4); }
        .btn-ghost { background: rgba(255,255,255,0.05); color: var(--text); border: 1px solid var(--border); box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .btn-ghost:hover { background: rgba(255,255,255,0.1); border-color: rgba(255,255,255,0.2); }
        
        .response-box {
            background: #000;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            font-family: var(--mono);
            font-size: 13px;
            color: var(--text);
            max-height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-all;
            line-height: 1.7;
            min-height: 120px;
            box-shadow: inset 0 0 20px rgba(0,0,0,0.8);
        }
        .response-box.empty {
            color: var(--hint);
            font-family: var(--font);
            font-size: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            gap: 12px;
            background: rgba(0,0,0,0.3);
        }
        .response-box.empty::before {
            content: 'Terminal Output';
            font-family: var(--mono);
            font-size: 12px;
            opacity: 0.5;
            letter-spacing: 2px;
            text-transform: uppercase;
        }

        .response-meta {
            display: flex; align-items: center; gap: 12px; margin-bottom: 12px;
            font-size: 12px; font-family: var(--mono); color: var(--muted);
            background: rgba(15, 23, 42, 0.6); padding: 8px 16px; border-radius: 8px; border: 1px solid var(--border);
        }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
        .s200 { background: var(--green); box-shadow: 0 0 10px var(--green); }
        .s4xx { background: var(--red); box-shadow: 0 0 10px var(--red); }
        .s-pending { background: var(--amber); animation: pulse-green 1s infinite; box-shadow: 0 0 10px var(--amber); }

        /* ── JSON COLOURING ─────────────────────── */
        .j-key { color: #f472b6; }
        .j-str { color: #a78bfa; }
        .j-num { color: #fbbf24; }
        .j-bool { color: #34d399; }
        .j-null { color: #9ca3af; font-style: italic; }

        /* ── FEATURE GRID ───────────────────────── */
        .feat-card {
            display: flex; flex-direction: column; gap: 12px;
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.7));
        }
        .feat-icon {
            width: 48px; height: 48px; border-radius: 12px; display: flex; align-items: center;
            justify-content: center; font-size: 24px; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.1);
        }
        .feat-card h4 { font-size: 16px; font-weight: 700; color: #fff; }
        .feat-card p { font-size: 14px; color: var(--muted); line-height: 1.6; }

        /* ── SCHEMA TABLE ───────────────────────── */
        .schema-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 14px; }
        .schema-table th {
            text-align: left; padding: 16px; font-size: 12px; font-weight: 700; color: var(--hint);
            text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid var(--border);
            background: rgba(0,0,0,0.2);
        }
        .schema-table th:first-child { border-top-left-radius: 12px; }
        .schema-table th:last-child { border-top-right-radius: 12px; }
        .schema-table td { padding: 16px; border-bottom: 1px solid var(--border); vertical-align: top; background: rgba(0,0,0,0.1); transition: background 0.2s; }
        .schema-table tr:hover td { background: rgba(255,255,255,0.03); }
        .schema-table tr:last-child td:first-child { border-bottom-left-radius: 12px; }
        .schema-table tr:last-child td:last-child { border-bottom-right-radius: 12px; }
        .schema-table .field-name { font-family: var(--mono); font-weight: 600; color: var(--blue); }
        .schema-table .field-type { font-family: var(--mono); font-size: 12px; color: var(--purple); background: var(--purple-bg); padding: 2px 8px; border-radius: 6px; white-space: nowrap; }
        .schema-table .field-desc { color: var(--text); }

        /* ── ACTION SCHEMA ─────────────────────── */
        .action-block {
            background: rgba(0,0,0,0.2); border: 1px solid var(--border); border-radius: 12px; padding: 16px; margin-bottom: 12px; transition: all 0.2s;
        }
        .action-block:hover { background: rgba(255,255,255,0.03); border-color: rgba(255,255,255,0.1); transform: translateX(4px); }
        .action-block .action-name {
            font-family: var(--mono); font-size: 14px; color: var(--green); font-weight: 700; margin-bottom: 8px; display: inline-block; background: var(--green-bg); padding: 2px 10px; border-radius: 6px;
        }
        .action-block .action-params { font-size: 13px; color: var(--muted); font-family: var(--mono); background: #000; padding: 8px 12px; border-radius: 8px; margin-top: 8px; border: 1px solid rgba(255,255,255,0.05); }

        code { font-family: var(--mono); font-size: 13px; color: #fff; background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.1); }

        /* ── ENDPOINT LIST ──────────────────────── */
        .endpoint-row {
            display: flex; align-items: center; gap: 20px; padding: 20px; border-bottom: 1px solid var(--border); transition: all 0.2s;
        }
        .endpoint-row:hover { background: rgba(255,255,255,0.02); }
        .endpoint-row:last-child { border-bottom: none; }
        .method {
            font-size: 13px; font-weight: 800; font-family: var(--mono); padding: 6px 12px; border-radius: 8px; min-width: 60px; text-align: center; flex-shrink: 0; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.1);
        }
        .GET { background: var(--green-bg); color: var(--green); }
        .POST { background: var(--blue-bg); color: var(--blue); }
        .WS { background: var(--purple-bg); color: var(--purple); }
        .endpoint-path { font-family: var(--mono); font-size: 16px; font-weight: 600; color: #fff; margin-bottom: 4px; }
        .endpoint-desc { font-size: 14px; color: var(--muted); }

        /* ── FOOTER ─────────────────────────────── */
        footer {
            border-top: 1px solid var(--border); background: rgba(3, 7, 18, 0.8); backdrop-filter: blur(20px);
            padding: 40px; text-align: center; font-size: 14px; color: var(--hint);
        }
        footer strong { color: #fff; letter-spacing: 0.5px; }
        footer a { color: var(--blue); text-decoration: none; font-weight: 500; transition: color 0.2s; }
        footer a:hover { color: #fff; text-shadow: 0 0 10px var(--blue); }

        /* ── SPINNER & COPY BTN ─────────────────── */
        .spinner {
            display: inline-block; width: 16px; height: 16px; border: 2px solid rgba(255,255,255,0.2);
            border-top-color: #fff; border-radius: 50%; animation: spin 0.8s linear infinite; margin-right: 8px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .copy-btn {
            position: absolute; top: 16px; right: 16px; background: rgba(255,255,255,0.1); border: none;
            color: #fff; border-radius: 6px; padding: 6px 12px; font-size: 12px; font-weight: 600; cursor: pointer;
            transition: all 0.2s; backdrop-filter: blur(4px); text-transform: uppercase; letter-spacing: 0.5px;
        }
        .copy-btn:hover { background: rgba(255,255,255,0.2); transform: scale(1.05); }
        .copy-btn:active { transform: scale(0.95); }
        .relative { position: relative; }

        /* ── STATUS BAR ─────────────────────────── */
        .status-bar {
            background: linear-gradient(90deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.8));
            border: 1px solid var(--border); border-radius: 12px; padding: 16px 24px;
            display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 20px;
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.2);
        }
        .status-item { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--muted); font-family: var(--mono); }
        .status-item strong { color: #fff; background: rgba(0,0,0,0.3); padding: 4px 8px; border-radius: 6px; font-size: 14px; }

        /* ── LEGEND ─────────────────────────────── */
        .legend { display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 24px; padding: 16px; background: rgba(0,0,0,0.2); border-radius: 12px; border: 1px solid var(--border); }
        .legend-item { display: flex; align-items: center; gap: 8px; font-size: 13px; font-weight: 600; color: var(--text); }
        .legend-swatch { width: 16px; height: 16px; border-radius: 4px; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.2); }
        
        [title] { cursor: help; border-bottom: 1px dotted rgba(255,255,255,0.3); }
        
        .chip-container { display: flex; flex-wrap: wrap; gap: 6px; }
"""

head_end = content.find("</style>")
if head_end == -1:
    print("Could not find </style> end tag.")
    exit(1)

fonts_link = """
    <!-- Google Fonts for Modern Aesthetics -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
"""

start_pos = content.find("<style>")
if start_pos == -1:
    print("Could not find <style> tag.")
    exit(1)

prefix = content[:start_pos]

body_content = content[head_end+8:]

body_content = body_content.replace(
"""      <div class="task-footer">
        ${t.chips.map(c => `<span class="chip ${chipColor(c)}">${c.replace(/_/g, ' ')}</span>`).join('')}
      </div>""",
"""      <div class="task-footer chip-container">
        ${t.chips.map(c => `<span class="chip ${chipColor(c)}">${c.replace(/_/g, ' ')}</span>`).join('')}
      </div>"""
)

final_content = prefix + fonts_link + "<style>\n" + new_css + "\n</style>" + body_content

with open(filepath, "w", encoding="utf-8") as f:
    f.write(final_content)

print("Redesign applied successfully.")