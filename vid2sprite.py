#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vid2sprite.py — 绿幕视频 → 游戏可用精灵帧

与 vid2sprite.html 的关系：HTML 版是本脚本的浏览器移植（零依赖、双击即用），
核心算法（chroma_alpha / unpremultiply / fill_holes / largest_component）
两边保持同一实现，改一边记得同步另一边。HTML 版位于同目录。

三个阶段，可单独跑：
  frames  视频 → PNG 序列（抽帧 + 去重复帧）
  key     PNG 序列 → 带 alpha 的 PNG（抠绿幕 + 去溢色 + 去碎屑 + 裁剪）
  strip   PNG 序列 → 横向条带（烘焙成 1 张图，运行时 1 次 drawImage）
  selftest 自检：用已知 ground-truth 的合成图验证抠像质量

依赖：numpy / Pillow / imageio-ffmpeg（仅需 frames 阶段）
用法：
  python vid2sprite.py frames  in.mp4  out/ --fps 12
  python vid2sprite.py key     out/    out_key/
  python vid2sprite.py strip   out_key/ art/vfx/v2/xxx_walk.png
  python vid2sprite.py selftest

关键设计（why 都写在注释里）：
  1. alpha 用"相对绿超出度"而不是绝对差值 —— 屏幕有明暗渐变时绝对差值会误杀暗部。
  2. 抠像后必须做 un-premultiply 还原真实颜色 —— 这是"去溢色"的正解，
     比"把 G 往 max(R,B) 压"之类的经验 despill 更干净，且能救回运动模糊的半透明边缘。
  3. alpha 先腐蚀 1px 再输出 —— 残留绿边在精灵尺寸下是可见的绿描边。
"""

import os
import sys
import math
import subprocess
import argparse
from collections import deque

import numpy as np
from PIL import Image, ImageFilter

# ---------------------------------------------------------------- 抠像核心

def smoothstep(lo, hi, x):
    """平滑阶跃。why: 硬阈值会在边缘产生锯齿状 alpha；平滑过渡能保留运动模糊的半透明。"""
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def estimate_screen(rgb, ring=8):
    """
    从画面外圈估计幕布颜色。
    why: 实拍绿幕几乎不可能是纯 (0,255,0)，打光不均时中心和外圈能差 30+。
    用外圈中位数比硬编码纯绿鲁棒得多。
    """
    h, w, _ = rgb.shape
    ring = max(2, min(ring, h // 4, w // 4))
    border = np.concatenate([
        rgb[:ring].reshape(-1, 3),
        rgb[-ring:].reshape(-1, 3),
        rgb[:, :ring].reshape(-1, 3),
        rgb[:, -ring:].reshape(-1, 3),
    ]).astype(np.float64)
    exc = border[:, 1] - np.maximum(border[:, 0], border[:, 2])
    sel = border[exc > 25.0]
    if len(sel) < 64:
        return np.array([0.0, 255.0, 0.0])   # 外圈绿得不够 → 退回纯绿
    return np.median(sel, axis=0)


def chroma_alpha(rgb, screen, lo=0.12, hi=0.35, abs_lo=0.0, abs_hi=0.0):
    """
    由色度算 alpha。
    绿超出度 = (G - max(R,B)) / G，归一化的好处是暗部屏幕也能识别
    （绝对差值下 (10,60,10) 只超出 50，会被当成前景）。

    abs_lo/abs_hi（绝对绿超出门控）：
    相对化有个反噬 —— 暗色像素 G 小、分母小，一点点绿就被放大成"很绿"，
    实测把绿幕前的黑发抠出成片的洞（洞像素 G-maxRB 仅 3.5~11，背景 20~22）。
    所以相对色度必须用绝对色度兜底：绝对绿超出低于 abs_lo 的像素绝不判为背景。
    0 = 关闭门控（兼容旧行为）。
    """
    f = rgb.astype(np.float64)
    g = f[:, :, 1]
    mx = np.maximum(f[:, :, 0], f[:, :, 2])
    abs_exc = g - mx
    excess = abs_exc / np.maximum(g, 1.0)
    # 幕布本身不是纯绿时（比如 (30,200,40)），excess 会偏小 → 用幕布的 excess 做基准归一
    kexc = (screen[1] - max(screen[0], screen[2])) / max(screen[1], 1.0)
    if kexc > 1e-6:
        excess = excess / kexc
    if abs_hi > abs_lo > 0:
        # 门控 = 「绝对绿超出够高」**或**「亮度够高」→ 判背景；
        # 两者都低（暗发丝：abs_exc 低且远暗于幕布）→ 救回。
        # why 用 max 不用乘：乘会把亮幕布自己的 excess 一并清零，整块背景被保留（实测）。
        # 亮度项同时排掉烟雾（白烟+幕布混合：abs_exc 低但亮度高于幕布）。
        g_lo = 0.65 * screen[1]
        g_hi = 0.85 * screen[1]
        gate = np.maximum(smoothstep(abs_lo, abs_hi, abs_exc),
                          smoothstep(g_lo, g_hi, g))
        excess = excess * gate
    return 1.0 - smoothstep(lo, hi, excess)


def fill_holes(alpha, rgb_raw, thresh=0.5):
    """
    填掉被前景完全包围的透明洞（从边界 flood fill，到不了的透明区就是洞）。
    why: 发丝间偶尔残留小洞，外轮廓不受影响。颜色用原始观测值 ——
    包围洞里的是头发本体，不是前景/幕布混合，不能用 un-premultiply 的结果。
    与头发丝之间真正通向外界的缝隙无关（那些 reachable，不会被填）。
    """
    m = alpha > thresh
    h, w = m.shape
    outside = np.zeros_like(m)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if not m[y, x] and not outside[y, x]:
                outside[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if not m[y, x] and not outside[y, x]:
                outside[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not m[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                q.append((ny, nx))
    hole = (~m) & (~outside)
    if not hole.any():
        return alpha, rgb_raw
    a = alpha.copy()
    a[hole] = 1.0
    out = rgb_raw.copy()
    out[hole] = rgb_raw[hole]
    return a, out


def unpremultiply(rgb, alpha, screen, amin=0.12):
    """
    C_obs = a*C_fg + (1-a)*C_screen  →  C_fg = (C_obs - (1-a)*C_screen) / a
    why: 这一步同时干了"抠"和"去溢色"两件事。半透明边缘（运动模糊/毛发）
    本质是前景与幕布的线性混合，直接减掉幕布贡献就能还原真实颜色，
    而不是靠"把 G 压下去"那种会弄脏高光的经验做法。
    """
    f = rgb.astype(np.float64)
    a = alpha.astype(np.float64)
    mask = a > amin
    safe = np.where(mask, a, 1.0)[..., None]
    fg = (f - (1.0 - a[..., None]) * screen[None, None, :]) / safe
    fg = np.clip(fg, 0.0, 255.0)
    a = np.where(mask, a, 0.0)
    return fg.astype(np.uint8), a


def largest_component(alpha, thresh=0.5):
    """
    只保留最大连通块，扔掉幕布上的噪点/碎屑。
    why: 实拍素材几乎一定有机身反光、地面反光之类的小绿块会被误判成前景。
    """
    m = alpha > thresh
    if not m.any():
        return alpha
    h, w = m.shape
    seen = np.zeros_like(m)
    best = None
    best_n = 0
    for sy in range(h):
        for sx in range(w):
            if not m[sy, sx] or seen[sy, sx]:
                continue
            comp = []
            q = deque([(sy, sx)])
            seen[sy, sx] = True
            while q:
                y, x = q.pop()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if len(comp) > best_n:
                best_n = len(comp)
                best = comp
    out = np.zeros_like(alpha)
    for y, x in best:
        out[y, x] = alpha[y, x]
    return out


def key_image(img, lo=0.12, hi=0.35, erode=1, keep_largest=True, abs_lo=0.0, abs_hi=0.0, fill=False):
    """PIL RGB(A) → PIL RGBA。返回 (结果图, 幕布色)。"""
    rgb = np.array(img.convert('RGB'))
    screen = estimate_screen(rgb)
    a = chroma_alpha(rgb, screen, lo, hi, abs_lo, abs_hi)
    fg, a = unpremultiply(rgb, a, screen)
    if erode > 0:
        am = Image.fromarray((a * 255).astype(np.uint8), mode='L')
        for _ in range(erode):
            am = am.filter(ImageFilter.MinFilter(3))   # 腐蚀：吃掉残留绿边
        a = np.array(am).astype(np.float64) / 255.0
    if fill:
        a, fg = fill_holes(a, rgb)                     # 补洞：救回被误抠的内部区域
    if keep_largest:
        a = largest_component(a)
    out = np.dstack([fg, (a * 255).astype(np.uint8)])
    return Image.fromarray(out, mode='RGBA'), screen


def trim_to_alpha(img, pad=0):
    """裁到 alpha 包围盒。why: 条带格子尺寸直接决定纹理内存和屏幕占比。"""
    a = np.array(img.split()[-1])
    ys, xs = np.where(a > 8)
    if len(xs) == 0:
        return img
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + 1 + pad, img.width)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + 1 + pad, img.height)
    return img.crop((x0, y0, x1, y1))


# ---------------------------------------------------------------- 视频拆帧

def find_ffmpeg():
    exe = os.environ.get('FFMPEG')
    if exe and os.path.exists(exe):
        return exe
    try:
        import imageio_ffmpeg
        p = imageio_ffmpeg.get_ffmpeg_exe()
        if os.path.exists(p):
            return p
    except Exception:
        pass
    for c in ('ffmpeg', 'ffmpeg.exe'):
        from shutil import which
        p = which(c)
        if p:
            return p
    return None


def cmd_frames(args):
    ff = find_ffmpeg()
    if not ff:
        print('[FAIL] 找不到 ffmpeg。装上：pip install imageio-ffmpeg')
        return 2
    os.makedirs(args.out, exist_ok=True)
    vf = []
    if args.fps:
        vf.append('fps=%g' % args.fps)
    if args.scale:
        vf.append('scale=%d:-1' % args.scale)
    cmd = [ff, '-y', '-i', args.src]
    if vf:
        cmd += ['-vf', ','.join(vf)]
    cmd += ['-start_number', '0', os.path.join(args.out, '%04d.png')]
    print('>', ' '.join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        return r.returncode
    n = len([f for f in os.listdir(args.out) if f.lower().endswith('.png')])
    print('[OK] %d 帧 → %s' % (n, args.out))
    if args.dedupe > 0:
        return dedupe(args.out, args.dedupe)
    return 0


def dedupe(d, thresh):
    """
    丢掉和上一保留帧几乎一样的帧。
    why: 视频里大量静止片段，全抽进来会让条带塞满重复姿态
    —— 咱们吃过这亏：4 帧里只有 3 个不同姿态，帧数不等于姿态数。

    判定用「变化像素占比」而不是「整帧平均差」：
    整帧平均差在主体只占画面一小块时会自相消（实测 12 帧被丢到只剩 1 帧），
    和 gait-check 那条包围盒自相消是同一类错误。占比是尺度无关的。
    """
    fs = sorted(f for f in os.listdir(d) if f.lower().endswith('.png'))
    prev = None
    kept, dropped = [], 0
    for f in fs:
        a = np.array(Image.open(os.path.join(d, f)).convert('L')).astype(np.float64)
        if prev is not None and a.shape == prev.shape:
            frac = float((np.abs(a - prev) > 10).mean())
            if frac < thresh:
                os.remove(os.path.join(d, f))
                dropped += 1
                continue
        prev = a
        kept.append(f)
    print('[OK] 去重复：保留 %d，丢弃 %d（变化像素占比阈值 %.3f）' % (len(kept), dropped, thresh))
    return 0


# ---------------------------------------------------------------- 阶段命令

def list_pngs(d):
    if os.path.isfile(d):
        return [d]
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith('.png'))


def cmd_key(args):
    srcs = list_pngs(args.src)
    if not srcs:
        print('[FAIL] 没有 PNG：', args.src)
        return 2
    os.makedirs(args.out, exist_ok=True)
    screens = []
    for p in srcs:
        im = Image.open(p).convert('RGB')
        out, sc = key_image(im, args.lo, args.hi, args.erode, not args.keep_all,
                            args.abs_gate, args.abs_gate + 8.0, args.fill_holes)
        if args.trim:
            out = trim_to_alpha(out, args.pad)
        out.save(os.path.join(args.out, os.path.basename(p)))
        screens.append(sc)
    sc = np.median(np.array(screens), axis=0)
    print('[OK] %d 张 → %s' % (len(srcs), args.out))
    print('     幕布色估计 RGB=(%.0f, %.0f, %.0f)' % tuple(sc))
    if sc[1] - max(sc[0], sc[2]) < 40:
        print('[WARN] 幕布色不够绿 —— 素材可能不是绿幕，或外圈取样失败')
    return 0


def cmd_strip(args):
    srcs = list_pngs(args.src)
    if not srcs:
        print('[FAIL] 没有 PNG：', args.src)
        return 2
    ims = [Image.open(p).convert('RGBA') for p in srcs]
    cw = max(i.width for i in ims)
    ch = max(i.height for i in ims)
    if args.cell:
        ch = args.cell
        cw = int(round(ch * max(i.width / i.height for i in ims)))
    sheet = Image.new('RGBA', (cw * len(ims), ch), (0, 0, 0, 0))
    for i, im in enumerate(ims):
        if args.cell:
            k = args.cell / im.height
            im = im.resize((max(1, int(round(im.width * k))), args.cell), Image.LANCZOS)
        sheet.paste(im, ((cw - im.width) // 2 + i * cw, (ch - im.height) // 2), im)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sheet.save(args.out)
    print('[OK] 条带 %dx%d，%d 格（每格 %dx%d）→ %s' % (sheet.width, sheet.height, len(ims), cw, ch, args.out))
    print('     游戏内接入：BUG_WALK[\'xxx\'] = {file:\'%s\', n:%d}' % (args.out, len(ims)))
    return 0


def cmd_preview(args):
    """
    拼一张对照图：上行=原图（绿幕），下行=抠像后合成到游戏深色背景（--zoom 倍）。
    why: 抠像是最容易"跑完没报错但其实全错"的活，必须一眼能验收。
    而且必须**放大和 1:1 两种尺寸都看** —— 1px 绿边在缩略图上是亚像素，放大才会露馅。
    """
    srcs = list_pngs(args.src)
    if not srcs:
        print('[FAIL] 没有 PNG：', args.src)
        return 2
    ims = [Image.open(p).convert('RGBA') for p in srcs]
    z = args.zoom
    cw = max(i.width for i in ims) * z
    ch = max(i.height for i in ims) * z
    cols = min(args.cols, len(ims))
    rows = (len(ims) + cols - 1) // cols
    H = rows * ch * 2 + 24
    sheet = Image.new('RGB', (cols * cw, H), (24, 22, 26))
    for k, im in enumerate(ims):
        r, c = divmod(k, cols)
        x, y = c * cw, r * ch * 2
        raw = Image.new('RGB', (im.width, im.height), (0, 170, 0))
        raw.paste(im.convert('RGB'), (0, 0), im)
        sheet.paste(raw.resize((im.width * z, im.height * z), Image.NEAREST), (x, y))
        bg = Image.new('RGB', (im.width, im.height), (26, 24, 30))
        bg.paste(im, (0, 0), im)
        sheet.paste(bg.resize((im.width * z, im.height * z), Image.NEAREST), (x, y + ch))
    sheet.save(args.out)
    print('[OK] 预览 %dx%d（上=原图 下=抠像合成，%d×）→ %s' % (sheet.width, sheet.height, z, args.out))
    return 0


# ---------------------------------------------------------------- 自检

def cmd_selftest(args):
    """
    合成一张已知 alpha 的绿幕图，跑完整抠像，和 ground truth 比。
    why: 抠像是最容易"看起来还行其实全错"的活 —— 必须量化验证，不能目测。
    """
    n = 256
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    cx = cy = n / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # ground truth alpha：中心实心 + 外圈 8px 软边（模拟运动模糊）
    a_gt = np.clip((72.0 - r) / 8.0, 0.0, 1.0)
    # ground truth 前景色：黄褐色甲壳（故意带一点绿，模拟难抠的情况）
    fg_gt = np.zeros((n, n, 3))
    fg_gt[:, :, 0] = 190 + 20 * np.sin(xx / 17.0)
    fg_gt[:, :, 1] = 150 + 18 * np.cos(yy / 13.0)
    fg_gt[:, :, 2] = 90 + 12 * np.sin((xx + yy) / 21.0)
    screen = np.array([10.0, 210.0, 30.0])   # 非纯绿，模拟真实打光
    obs = fg_gt * a_gt[..., None] + screen[None, None, :] * (1.0 - a_gt[..., None])
    obs = np.clip(obs, 0, 255).astype(np.uint8)

    got, sc_est = key_image(Image.fromarray(obs, mode='RGB'))
    a_got = np.array(got.split()[-1]).astype(np.float64) / 255.0
    rgb_got = np.array(got.convert('RGB')).astype(np.float64)

    err = np.abs(a_got - a_gt).mean()
    core = a_gt > 0.95
    col_err = np.abs(rgb_got - np.clip(fg_gt, 0, 255))[core].mean() if core.any() else 999.0
    sc_err = np.abs(sc_est - screen).max()

    print('alpha 平均绝对误差 : %.4f  (阈值 < 0.05)' % err)
    print('核心区颜色误差     : %.2f  (阈值 < 8.0)' % col_err)
    print('幕布色估计误差     : %.2f  (阈值 < 12.0)' % sc_err)

    ok = True
    if err >= 0.05:
        print('[FAIL] alpha 误差超标'); ok = False
    if col_err >= 8.0:
        print('[FAIL] 去溢色不干净，核心区颜色偏了'); ok = False
    if sc_err >= 12.0:
        print('[FAIL] 幕布色估计不准'); ok = False
    # 负向对照：幕布纯色处 alpha 必须是 0
    corner = a_got[:6, :6].max()
    if corner > 0.02:
        print('[FAIL] 幕布角落没抠干净 alpha=%.3f' % corner); ok = False
    print('角落残留 alpha    : %.4f' % corner)

    # 用例 B：深色主体 + 内部绿灰高光斑 —— 相对色度把高光斑误判成背景，抠出"洞"
    # （本项目实测：黑发整体保住，但发间的绿灰高光 (73,84,65) 被抠穿）
    fg2 = np.zeros((n, n, 3))
    fg2[:, :, 0] = 60; fg2[:, :, 1] = 62; fg2[:, :, 2] = 58   # 中性深灰"头发"（应保留）
    inner = np.sqrt((xx - (cx + 14)) ** 2 + (yy - (cy - 10)) ** 2) < 26.0
    fg2[inner] = (73, 84, 65)                                  # 绿灰"发丝高光"（无门控会被抠成洞）
    m2 = np.clip((70.0 - r) / 6.0, 0.0, 1.0)
    # 必须用暗调幕布（实测素材是 (114,141,105)）：暗幕布 kexc≈0.21，
    # 相对色度被放大 ~5 倍才触发这个 bug；纯亮绿幕 kexc≈0.95 复现不出来
    screen2 = np.array([110.0, 140.0, 105.0])
    # 烟雾斑（白烟 70% + 幕布 30% 的混合观测色）：abs_exc 低、亮度高，
    # 只救暗部的门控必须把它排除，否则留下灰绿雾块
    smoke = np.sqrt((xx - (cx - 60)) ** 2 + (yy - (cy - 55)) ** 2) < 22.0
    obs2 = fg2 * m2[..., None] + screen2[None, None, :] * (1 - m2[..., None])
    sm = 0.4 * np.array([255.0, 255.0, 255.0]) + 0.6 * screen2   # (168,186,165) 淡烟
    # 注：更浓的烟（如 76% 白）任何绿度键控都无解 —— 那是"主体含键控色"问题，
    # 得换幕布色或走 AI matting，不是阈值能救的
    obs2 = obs2 * (~smoke)[..., None] + sm[None, None, :] * smoke[..., None]
    obs2 = np.clip(obs2, 0, 255).astype(np.uint8)
    # 不开门控：应出现大量洞（复现 bug）
    bad_img, _ = key_image(Image.fromarray(obs2, mode='RGB'), erode=0, keep_largest=False)
    holes_bad = count_holes(np.array(bad_img.split()[-1]) > 128)
    area_nogate = (np.array(bad_img.split()[-1]) > 128).sum() / max((m2 > 0.5).sum(), 1)
    # 开门控：洞应接近 0；且面积不得超过无门控 —— 不变量：门控只救暗部，绝不多留
    good_img, _ = key_image(Image.fromarray(obs2, mode='RGB'), erode=0, keep_largest=False, abs_lo=12.0, abs_hi=20.0, fill=True)
    ag = np.array(good_img.split()[-1]).astype(np.float64) / 255.0
    holes_good = count_holes(ag > 0.5)
    area_err = abs((ag > 0.5).sum() / max((m2 > 0.5).sum(), 1) - 1.0)
    print('深色主体：无门控洞像素 %d（应 > 200，证明 bug 存在）' % holes_bad)
    print('深色主体：开门控洞像素 %d（应 = 0），面积误差 %.3f（无门控为 %.3f）'
          % (holes_good, area_err, area_nogate))
    if holes_bad <= 200:
        print('[WARN] 无门控时没复现洞 —— 测试用例可能已失效')
    if holes_good > 0:
        print('[FAIL] 门控后仍有洞'); ok = False
    # 烟斑中心（亮部低abs_exc）必须被键掉 —— 门控只救暗部，不救亮部
    sy, sx = int(cy - 55 - 12), int(cx - 60 - 12)
    smoke_alpha = ag[sy:sy + 24, sx:sx + 24].mean()
    print('烟斑中心 alpha 均值: %.3f（应 < 0.5）' % smoke_alpha)
    if smoke_alpha > 0.5:
        print('[FAIL] 亮部烟雾被误救 —— 门控亮度项失效'); ok = False
    if area_err >= 0.15:
        print('[FAIL] 面积偏差过大 —— 主体被误抠'); ok = False

    print('[SELFTEST %s]' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


def count_holes(mask):
    """被前景包围的透明像素数（边界 flood fill）。"""
    from collections import deque as _dq
    h, w = mask.shape
    outside = np.zeros_like(mask)
    q = _dq()
    for x in range(w):
        for y in (0, h - 1):
            if not mask[y, x] and not outside[y, x]:
                outside[y, x] = True; q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if not mask[y, x] and not outside[y, x]:
                outside[y, x] = True; q.append((y, x))
    while q:
        y, x = q.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not mask[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True; q.append((ny, nx))
    return int(((~mask) & (~outside)).sum())


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description='绿幕视频 → 游戏精灵帧')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('frames', help='视频 → PNG 序列')
    p.add_argument('src'); p.add_argument('out')
    p.add_argument('--fps', type=float, default=0, help='抽帧率，0=全部')
    p.add_argument('--scale', type=int, default=0, help='缩放到指定宽度')
    p.add_argument('--dedupe', type=float, default=0.004, help='变化像素占比低于此值则丢弃该帧，0=关闭')
    p.set_defaults(fn=cmd_frames)

    p = sub.add_parser('key', help='PNG → 抠像 PNG')
    p.add_argument('src'); p.add_argument('out')
    p.add_argument('--lo', type=float, default=0.12, help='低于此绿超出度=完全不透明')
    p.add_argument('--hi', type=float, default=0.35, help='高于此绿超出度=完全透明')
    p.add_argument('--erode', type=int, default=1, help='alpha 腐蚀像素数，吃绿边')
    p.add_argument('--abs-gate', type=float, default=0.0,
                   help='绝对绿超出门控下限（G-max(R,B) 低于此值绝不当背景），0=关闭。'
                        '主体含深色/含绿时必开，建议 12')
    p.add_argument('--fill-holes', action='store_true', help='填充被前景包围的透明洞')
    p.add_argument('--keep-all', action='store_true', help='保留全部连通块（不去掉碎屑）')
    p.add_argument('--no-trim', dest='trim', action='store_false', help='不裁剪到包围盒')
    p.add_argument('--pad', type=int, default=0, help='裁剪后外扩像素')
    p.set_defaults(fn=cmd_key)

    p = sub.add_parser('strip', help='PNG 序列 → 横向条带')
    p.add_argument('src'); p.add_argument('out')
    p.add_argument('--cell', type=int, default=0, help='统一格子高度，0=用原尺寸')
    p.set_defaults(fn=cmd_strip)

    p = sub.add_parser('preview', help='拼对照图：原图 vs 抠像合成')
    p.add_argument('src'); p.add_argument('out')
    p.add_argument('--zoom', type=int, default=3)
    p.add_argument('--cols', type=int, default=6)
    p.set_defaults(fn=cmd_preview)

    p = sub.add_parser('selftest', help='抠像质量自检')
    p.set_defaults(fn=cmd_selftest)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == '__main__':
    sys.exit(main())
