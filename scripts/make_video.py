"""生成演示视频草稿：真实运行的命令（终端动画）+ 文档图表 + 中文语音旁白（macOS say）+ 字幕。

python scripts/make_video.py capture   # 真实运行演示命令，记录输出与耗时 → video/build/transcript.json
python scripts/make_video.py render    # 合成画面、旁白与字幕 → video/emslice_demo.mp4
python scripts/make_video.py all

依赖：ffmpeg（含 libass）、Pillow、macOS `say`（Tingting 语音）。终端中显示的耗时为实际运行耗时，视频中等待过程被压缩。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import wave

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "video")
BUILD = os.path.join(OUT, "build")
W, H = 1920, 1080
VOICE, RATE = "Tingting", 205

PINGFANG = ("/System/Library/AssetsV2/com_apple_MobileAsset_Font8/86ba2c91f017a3749571a82f2c6d890ac7ffb2fb.asset/"
            "AssetData/PingFang.ttc")
MENLO = "/System/Library/Fonts/Menlo.ttc"


def _font(path, size, index=0):
    try:
        return ImageFont.truetype(path, size, index=index)
    except OSError:
        return ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", size)


CJK = lambda s, w="Regular": _font(PINGFANG, s, {"Regular": 3, "Medium": 7, "Semibold": 11}[w])  # noqa: E731
MONO = lambda s: _font(MENLO, s, 0)  # noqa: E731

# ------------------------------------------------------------------ 分段脚本
# 旁白条目：字符串，或 (字幕文本, 朗读文本) 以修正专有名词读音
SEGMENTS = [
    dict(kind="card", title="emslice", sub="面向电磁谐振与微波器件仿真的非正定厄密稀疏特征值求解器",
         lines=["目标频段的谐振模式一个不漏，并给出可审计的证据"],
         foot="2026 上海开源软件应用创新大赛 · AI+工业软件赛道 · 中国商飞赛题 · Apache-2.0",
         say=[("大家好，这是 emslice，面向电磁谐振与微波器件仿真的大规模非正定厄密稀疏特征值求解器。",
               "大家好，这是 em slice，面向电磁谐振与微波器件仿真的大规模非正定厄密稀疏特征值求解器。"),
              "它要做的是：在十万阶以上的矩阵中，把目标频段的谐振模式一个不漏地算出来，并给出可以核查的证据。"]),
    dict(kind="card", title="问题：最怕的不是慢，而是漏模", sub="",
         lines=["• 滤波器、天线、环行器设计 → 十万阶稀疏厄密特征值问题（零空间与位移使矩阵非正定）",
                "• 漏掉一个寄生模，滤波器就可能在带内出现意外谐振",
                "• 现有开源工具只保证“残差小”，不保证“没有漏”",
                "• 目标频率恰为谐振点、重特征值、低频零空间：常规用法给出错误甚至空结果，且不报错"],
         say=["微波滤波器、天线、环行器的设计，都要计算结构的谐振频率与模式。",
              "工程上最怕的不是慢，而是漏掉模式：漏掉一个寄生模，滤波器就可能在带内出现意外的谐振。",
              "常用的开源工具只保证求出来的是特征对，不保证没有遗漏。",
              "在目标频率恰好落在谐振点、重特征值、低频零空间这些场景下，它们会给出错误甚至空的结果，而且不报错。"]),
    dict(kind="image", path="docs/arch.png", caption="架构：求解 — 核对 — 补解 闭环",
         say=["emslice 的核心是一条求解、核对、补解的闭环。",
              ("位移-求逆块 Krylov-Schur 负责求解，Sylvester 惯性定理负责核对：",
               "位移求逆块克雷洛夫舒尔方法负责求解，西尔维斯特惯性定理负责核对："),
              "对称不定分解的负主元个数，恰好等于小于该位移的特征值个数，与迭代过程完全独立。",
              "计数与求得个数一致，就证明窗口内没有遗漏；不一致则按缺口自动补解。"]),
    dict(kind="term", id="mid", title="场景一：目标频率恰为 6 重谐振点（十万阶立方腔）",
         cmds=["emslice gen cube --N 33 --out data/demo/cube33.npz",
               "emslice solve data/demo/cube33.npz --sigma 3306.3592973618242 -k 20 --out result/demo/mid --threads 8",
               "python scripts/show_report.py result/demo/mid"],
         say=["先看一个十万阶的立方谐振腔，它有大量六重、十二重的简并模式。",
              "我们把目标频率恰好放在一个六重谐振点上，此时位移矩阵是奇异的。",
              "emslice 自动检测近奇异位移，返回二十个特征值，后向误差约十的负十一次方；窗口内的惯性计数与求得个数一致，证书通过。"]),
    dict(kind="table", title="同一算例的对比（MacBook Air M2 基准，n ≈ 10 万，见 results/full/report.md）",
         header=["求解器", "耗时", "最大后向误差", "结果"],
         rows=[["emslice（含证书）", "13.9 s", "3e-11", "正确，证书通过"],
               ["SLEPc Krylov-Schur + MUMPS", "5.3 s", "0.12", "精度崩溃，漏 2 个简并模式"],
               ["SciPy eigsh + SuperLU", "386 s", "4e-9", "慢 28 倍"]],
         bad=[1],
         say=[("同一个问题，SLEPc 的常规用法后向误差达到零点一二，还漏掉了两个简并模式，并且没有任何报错。",
               "同一个问题，斯莱普西的常规用法后向误差达到零点一二，还漏掉了两个简并模式，并且没有任何报错。")]),
    dict(kind="term", id="low", title="场景二：低频击穿 — 3.2 万维梯度零空间挡在目标前",
         cmds=["emslice solve data/demo/cube33.npz --sigma 1.97e-5 --which above -k 20 --out result/demo/low --threads 8",
               "python scripts/show_report.py result/demo/low"],
         say=["第二个场景是低频击穿：三万两千多维的梯度零空间挡在目标前面。",
              "emslice 先用惯性计数把目标括在一个区间里，再把位移移到区间中部求解，得到前二十个物理模式，证书通过。",
              ("同样的问题，SLEPc 按常规做法返回零个有效结果。", "同样的问题，斯莱普西按常规做法返回零个有效结果。")]),
    dict(kind="image", path="docs/fem_dr_modes.png", caption="有限元工程场景：介质谐振器滤波腔（广义问题 Kx = k₀²Mx）",
         say=["为了贴近真实工程，项目自带有限元生成器：用 gmsh 生成非结构四面体网格，用棱边有限元离散，得到广义特征值问题。",
              "这是一个介质谐振器滤波腔：介电常数三十八的陶瓷圆柱放在支撑上，外面是铜腔。图中是求得的前两阶模的电场分布。"]),
    dict(kind="term", id="cav", title="场景三：十万阶有限元广义问题（空腔，与解析解对照）",
         cmds=["emslice gen fem-cavity --out data/demo/cav.npz",
               "emslice solve data/demo/cav.npz --B data/demo/cav_B.npz --sigma 1e-6 --which above -k 12 --out result/demo/cav --threads 8",
               "python scripts/show_report.py result/demo/cav --ghz",
               "python -c \"from emslice.femgen import *; print('解析解 (GHz)：', freq_ghz(analytic_cavity((30, 24, 18), 0.12))[:12].round(4))\""],
         say=["这是一个十万阶的有限元广义问题，同样有一万三千多维的梯度零空间。",
              "emslice 求出前十二个谐振频率，证书通过；与连续问题的解析解逐一对应，前二十阶频率误差不超过千分之一点四。",
              "前面图中的介质谐振器滤波腔，主模是三点三六吉赫兹，其后是寄生模。"]),
    dict(kind="image", path="docs/fem_time_compare.png", caption="有限元算例耗时对比（红色斜线 = 结果错误、为空或超时）",
         say=["这是有限元算例的对比。深蓝色是 emslice，带证书的耗时也低于 SLEPc。",
              "在低频场景，SLEPc 返回空结果，SciPy 精度降到十的负六次方；复厄密的铁氧体算例，SciPy 二十五分钟内没有算完。"]),
    dict(kind="image", path="docs/scaling_kunpeng.png", caption="华为鲲鹏 + openEuler：多核扩展性（单 NUMA 节点 1–32 线程）",
         say=["我们在华为鲲鹏服务器和欧拉操作系统上做了多核测试。",
              "十万阶复厄密算例从单线程五百二十五秒降到三十二线程三十三点七秒，加速十五点六倍，二十四次运行全部证书通过；同条件下 SLEPc 的结果是错误的。"]),
    dict(kind="term", id="check", title="独立复核与测试",
         cmds=["emslice check data/demo/dr.npz result/demo/dr --B data/demo/dr_B.npz",
               "python -m pytest -q"],
         say=["结果可以用 check 命令独立复核残差与正交性。",
              "项目有十八项单元与场景测试，持续集成覆盖 Linux 和 macOS 的完整后端，以及没有 PETSc 时的降级模式。"]),
    dict(kind="card", title="开源与可复现", sub="",
         lines=["• Apache-2.0；核心依赖 PETSc / MUMPS / METIS / NumPy / SciPy 均为宽松开源协议",
                "• GitHub Actions 持续集成 · Dockerfile · scripts/reproduce.sh 一键复现全部基准",
                "• 测试矩阵确定性生成、逐字节可复现；贡献指南、安全策略、路线图",
                "• 后续：混合精度、MPI 分布式谱切片、对接开源有限元软件、发布 PyPI / conda-forge"],
         say=["项目以 Apache 2.0 协议开源，提供持续集成、Docker 镜像和一键复现脚本，所有测试矩阵都可以确定性地重新生成。",
              "后续我们会加入混合精度和分布式谱切片，并与开源有限元软件对接。"]),
    dict(kind="card", title="emslice", sub="目标频段的谐振模式一个不漏，并给出可审计的证据", lines=[],
         foot="代码仓库 · 方案说明文档 · 作品介绍 见提交材料",
         say=[("emslice：目标频段的谐振模式一个不漏，并给出可审计的证据。谢谢观看。",
               "em slice：目标频段的谐振模式一个不漏，并给出可审计的证据。谢谢观看。")]),
]


# ------------------------------------------------------------------ 采集
def capture():
    os.makedirs(BUILD, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "4"), PYTHONUNBUFFERED="1")
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env["PATH"]
    tr = {}
    for seg in SEGMENTS:
        if seg["kind"] != "term":
            continue
        runs = []
        for cmd in seg["cmds"]:
            t0 = time.perf_counter()
            p = subprocess.run(cmd, shell=True, cwd=ROOT, env=env, capture_output=True, text=True)
            dt = time.perf_counter() - t0
            out = (p.stdout + p.stderr).rstrip("\n").splitlines()
            out = [ln for ln in out if not ln.startswith("  输出:")]          # 输出文件路径行过长，略去
            print(f"[{dt:6.1f}s] {cmd}  (exit {p.returncode}, {len(out)} 行)")
            if p.returncode != 0:
                raise SystemExit("\n".join(out[-20:]))
            runs.append(dict(cmd=cmd, out=out, seconds=dt))
        tr[seg["id"]] = runs
    with open(os.path.join(BUILD, "transcript.json"), "w", encoding="utf-8") as f:
        json.dump(tr, f, ensure_ascii=False, indent=1)


# ------------------------------------------------------------------ 绘制
BG, INK, ACCENT, MUTED, RED = (250, 251, 253), (24, 32, 48), (42, 111, 219), (110, 120, 135), (192, 57, 43)
CONTENT_BOTTOM = 930            # 下方留给字幕


def _is_cjk(ch):
    return ord(ch) > 0x2E7F or ch in "“”‘’…—•→≈≤≥·"


def draw_mixed(d, xy, text, size, fill, cjk_w="Regular"):
    """逐字选择字体：ASCII 用 Menlo（等宽），其余用苹方；返回结束 x。"""
    x, y = xy
    fm, fc = MONO(size), CJK(size, cjk_w)
    for ch in text:
        f = fc if _is_cjk(ch) else fm
        d.text((x, y), ch, font=f, fill=fill)
        x += f.getlength(ch)
    return x


def mixed_len(text, size):
    fm, fc = MONO(size), CJK(size)
    return sum((fc if _is_cjk(ch) else fm).getlength(ch) for ch in text)


def wrap(text, size, width):
    """按宽度折行，优先在空格处断开（避免把数字或路径拆成两半）。"""
    out, cur = [], ""
    for ch in text:
        if mixed_len(cur + ch, size) > width:
            k = cur.rfind(" ")
            if k > len(cur) // 3:
                out.append(cur[:k])
                cur = "  " + cur[k + 1:] + ch
            else:
                out.append(cur)
                cur = "  " + ch
        else:
            cur += ch
    out.append(cur)
    return out


def base():
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 8], fill=ACCENT)
    return im, d


def render_card(seg):
    im, d = base()
    big = seg["title"] == "emslice"
    if big:
        f = CJK(150, "Semibold")
        tw = f.getlength(seg["title"])
        d.text(((W - tw) / 2, 230), seg["title"], font=f, fill=ACCENT)
        y = 440
        for t, s, c in ((seg.get("sub"), 46, INK),) + tuple((ln, 38, MUTED) for ln in seg["lines"]):
            if t:
                f2 = CJK(s, "Medium")
                d.text(((W - f2.getlength(t)) / 2, y), t, font=f2, fill=c)
                y += s + 40
    else:
        d.text((120, 120), seg["title"], font=CJK(64, "Semibold"), fill=INK)
        d.rectangle([120, 215, 260, 221], fill=ACCENT)
        y = 290
        for ln in seg["lines"]:
            for part in wrap(ln, 40, W - 260):
                draw_mixed(d, (140, y), part, 40, INK)
                y += 64
            y += 18
    if seg.get("foot"):
        f3 = CJK(30)
        d.text(((W - f3.getlength(seg["foot"])) / 2, CONTENT_BOTTOM - 60), seg["foot"], font=f3, fill=MUTED)
    return im


def render_image(seg):
    im, d = base()
    d.text((100, 50), seg["caption"], font=CJK(48, "Semibold"), fill=INK)
    pic = Image.open(os.path.join(ROOT, seg["path"])).convert("RGB")
    bw, bh = W - 160, CONTENT_BOTTOM - 140
    s = min(bw / pic.width, bh / pic.height)
    pic = pic.resize((int(pic.width * s), int(pic.height * s)), Image.LANCZOS)
    im.paste(pic, ((W - pic.width) // 2, 130 + (bh - pic.height) // 2))
    return im


def render_table(seg):
    im, d = base()
    d.text((120, 110), seg["title"], font=CJK(52, "Semibold"), fill=INK)
    cols = [120, 820, 1020, 1300]
    y = 260
    for i, h in enumerate(seg["header"]):
        draw_mixed(d, (cols[i], y), h, 36, MUTED, "Medium")
    d.line([120, y + 70, W - 120, y + 70], fill=(200, 205, 215), width=2)
    y += 110
    for r, row in enumerate(seg["rows"]):
        c = RED if r in seg.get("bad", []) else INK
        if r == 0:
            d.rectangle([110, y - 18, W - 110, y + 70], fill=(223, 236, 252))
        for i, v in enumerate(row):
            draw_mixed(d, (cols[i], y), v, 36, c, "Medium" if r in seg.get("bad", []) or r == 0 else "Regular")
        y += 120
    return im


TERM_FONT, LINE_H = 27, 40
PANEL = (80, 130, W - 80, CONTENT_BOTTOM)


def term_frame(title, lines, cursor=False, host="emslice — conda activate emslice"):
    im, d = base()
    d.text((100, 40), title, font=CJK(46, "Semibold"), fill=INK)
    x0, y0, x1, y1 = PANEL
    d.rounded_rectangle([x0, y0, x1, y1], radius=18, fill=(22, 27, 36))
    d.rounded_rectangle([x0, y0, x1, y0 + 44], radius=18, fill=(48, 54, 66))
    d.rectangle([x0, y0 + 26, x1, y0 + 44], fill=(48, 54, 66))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([x0 + 22 + 30 * i, y0 + 14, x0 + 38 + 30 * i, y0 + 30], fill=c)
    draw_mixed(d, (x0 + 140, y0 + 9), host, 22, (170, 176, 188))
    maxl = (y1 - y0 - 70) // LINE_H
    show = lines[-maxl:]
    y = y0 + 60
    for kind, txt in show:
        col = {"cmd": (255, 214, 102), "out": (222, 226, 232), "note": (120, 200, 255), "ok": (130, 220, 150)}[kind]
        draw_mixed(d, (x0 + 28, y), txt, TERM_FONT, col)
        y += LINE_H
    if cursor:
        d.rectangle([x0 + 28 + mixed_len(show[-1][1], TERM_FONT) + 4, y - LINE_H + 6,
                     x0 + 28 + mixed_len(show[-1][1], TERM_FONT) + 18, y - 6], fill=(255, 214, 102))
    return im


def term_events(seg, runs, host=None):
    """返回 [(图像, 时长)]：逐字输入命令 → 运行中提示（实际耗时）→ 输出逐段出现。"""
    width = PANEL[2] - PANEL[0] - 60
    lines, ev = [], []
    for run in runs:
        cmd_lines = wrap("$ " + run["cmd"], TERM_FONT, width)
        typed = ""
        full = " ".join(cmd_lines)
        step = 4
        for k in range(step, len(full) + step, step):
            typed = full[:k]
            cur = wrap(typed, TERM_FONT, width)
            ev.append((term_frame(seg["title"], host=host or "emslice — conda activate emslice", lines=lines + [("cmd", c) for c in cur], cursor=True), 0.045))
        lines += [("cmd", c) for c in cmd_lines]
        if run["seconds"] > 1.5:
            ev.append((term_frame(seg["title"], host=host or "emslice — conda activate emslice", lines=lines + [("note", f"⏳ 运行中…（实际耗时 {run['seconds']:.1f} s，视频中压缩）")]), 1.3))
        outs = []
        for o in run["out"]:
            outs += wrap(o, TERM_FONT, width) if o else [""]
        kind = lambda t: "ok" if ("证书=通过" in t or "完备性证书：通过" in t or "passed" in t) else "out"  # noqa: E731
        for i in range(0, len(outs), 3):
            lines += [(kind(t), t) for t in outs[i:i + 3]]
            ev.append((term_frame(seg["title"], host=host or "emslice — conda activate emslice", lines=lines), 0.12))
        ev.append((term_frame(seg["title"], host=host or "emslice — conda activate emslice", lines=lines), 0.9))
    return ev


# ------------------------------------------------------------------ 旁白
def tts(text, path):
    aiff = path[:-4] + ".aiff"
    subprocess.run(["say", "-v", VOICE, "-r", str(RATE), "-o", aiff, text], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", aiff, "-ar", "44100", "-ac", "1", path], check=True)
    os.remove(aiff)
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def silence(path, sec):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                    "-t", f"{sec:.3f}", path], check=True)


def srt_time(t):
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"


def render():
    tr = json.load(open(os.path.join(BUILD, "transcript.json"), encoding="utf-8"))
    fr_dir, au_dir = os.path.join(BUILD, "frames"), os.path.join(BUILD, "audio")
    os.makedirs(fr_dir, exist_ok=True)
    os.makedirs(au_dir, exist_ok=True)
    concat, audio, subs = [], [], []
    t = 0.0
    nf = 0
    gap = 0.25
    for si, seg in enumerate(SEGMENTS):
        # 旁白：逐句合成，字幕随句切换
        say_t, items = 0.6, []
        for j, item in enumerate(seg["say"]):
            sub, spoken = (item, item) if isinstance(item, str) else item
            p = os.path.join(au_dir, f"s{si:02d}_{j:02d}.wav")
            dur = tts(spoken, p)
            items.append((p, dur, sub))
        if seg["kind"] == "term":
            ev = term_events(seg, tr[seg["id"]], tr.get("_host", {}).get(seg["id"]))
        else:
            im = {"card": render_card, "image": render_image, "table": render_table}[seg["kind"]](seg)
            ev = [(im, 0.1)]
        anim = sum(d for _, d in ev)
        speech = 0.6 + sum(d + gap for _, d, _ in items)
        total = max(anim + 0.8, speech + 0.7)
        ev[-1] = (ev[-1][0], ev[-1][1] + total - anim)          # 末帧停留到旁白结束
        for im, d in ev:
            fp = os.path.join(fr_dir, f"f{nf:05d}.png")
            im.save(fp)
            concat.append((fp, d))
            nf += 1
        # 音频：开头 0.6 s 静音，句间 0.25 s，末尾补齐
        p0 = os.path.join(au_dir, f"s{si:02d}_lead.wav")
        silence(p0, 0.6)
        audio.append(p0)
        cur = t + 0.6
        for p, dur, sub in items:
            audio.append(p)
            subs.append((cur, cur + dur, sub))
            pg = os.path.join(au_dir, f"gap_{si:02d}_{len(audio)}.wav")
            silence(pg, gap)
            audio.append(pg)
            cur += dur + gap
        pt = os.path.join(au_dir, f"s{si:02d}_tail.wav")
        silence(pt, total - speech)
        audio.append(pt)
        t += total
        print(f"段 {si:2d} {seg['kind']:5s} {total:5.1f} s（旁白 {speech:4.1f} s，画面 {anim:4.1f} s）")
    with open(os.path.join(BUILD, "frames.txt"), "w") as f:
        for fp, d in concat:
            f.write(f"file '{fp}'\nduration {d:.3f}\n")
        f.write(f"file '{concat[-1][0]}'\n")
    with open(os.path.join(BUILD, "audio.txt"), "w") as f:
        for p in audio:
            f.write(f"file '{p}'\n")
    with open(os.path.join(BUILD, "subs.srt"), "w", encoding="utf-8") as f:
        for i, (a, b, s) in enumerate(subs, 1):
            f.write(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{s}\n\n")
    narr = os.path.join(BUILD, "narration.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i",
                    os.path.join(BUILD, "audio.txt"), "-c", "copy", narr], check=True)
    style = ("FontName=PingFang SC,FontSize=13,PrimaryColour=&H00FFFFFF,"
             "BorderStyle=3,OutlineColour=&H99000000,Outline=1,Shadow=0,MarginV=14")
    srt = os.path.join(BUILD, "subs.srt").replace(":", r"\:")
    out = os.path.join(OUT, "emslice_demo.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i",
                    os.path.join(BUILD, "frames.txt"), "-i", narr,
                    "-vf", f"fps=25,subtitles='{srt}':force_style='{style}',format=yuv420p",   # 先定帧率，静态段每帧都有字幕
                    "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
                    "-shortest", "-movflags", "+faststart", out], check=True)
    print(out, f"{t:.1f} s")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("capture", "all"):
        capture()
    if what in ("render", "all"):
        render()
