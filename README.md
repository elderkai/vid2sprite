# vid2sprite

绿幕视频/图片 → 游戏精灵帧的一键处理工具。

## 包含什么

| 文件 | 说明 |
|---|---|
| `vid2sprite.html` | **浏览器单文件工具**，零依赖，双击即用（file:// 可运行，无网络请求） |
| `vid2sprite.py` | Python 命令行管线（frames / key / strip / preview / selftest） |

推荐直接用 HTML 版：拖入绿幕视频或图片即可出结果，不消耗任何 API token。

## 浏览器版功能（双模式页签）

- **抠视频**：选择/拖入绿幕视频 → 自动抽帧去重 → 逐帧抠幕 → 统一尺寸输出 PNG / ZIP / 条带图
- **抠图**：批量拖入绿幕图片 → 抠幕输出；支持**多主体模式**（差分表、表情表等一图多角色场景自动切块，按行聚类命名）

## 算法要点

- 相对绿超出度 `(G − max(R,B))/G` 做幕布归一色键
- **黑发门控**：暗幕布下相对色度被放大误抠深色主体，用「绝对绿超出 **或** 亮度阈值」救援判定
- un-premultiply 去溢色（消除边缘绿边）
- alpha erode 1px + 幕布色 ring 中位数估计 + 补洞
- 多主体：连通域保留 + `autoMinArea = max(64, 最大块×5%)` 自适应阈值 + 单主体防呆

## Python 版用法

```bash
pip install numpy pillow imageio-ffmpeg

# 抽帧
python vid2sprite.py frames 输入.mp4 -o out/frames

# 抠幕（深色主体加 --abs-gate 12 --fill-holes）
python vid2sprite.py key out/frames -o out/keyed --abs-gate 12 --fill-holes

# 合成条带
python vid2sprite.py strip out/keyed -o strip.png

# 自测
python vid2sprite.py selftest
```

## 输出

- 命名从 `1.png` 起连续编号，ZIP 内同样
- 输出几何三模式：full（原尺寸）/ bbox（裁包围盒）/ custom（统一等比缩放居中）
- 条带图 16384px 宽度上限，超出自动抽样

## License

MIT
