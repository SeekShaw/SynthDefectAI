"""生成应用图标 assets/app.ico：盾牌 + 芯片 + 缺陷划痕的极简设计。"""
import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "assets")
os.makedirs(OUT, exist_ok=True)

S = 256
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

# 圆角深色底
def rr(xy, r, fill):
    d.rounded_rectangle(xy, radius=r, fill=fill)

rr((12, 12, 244, 244), 52, (20, 24, 32, 255))
# 内边高光
d.rounded_rectangle((12, 12, 244, 244), radius=52, outline=(76, 154, 255, 90), width=3)

# 芯片主体
rr((64, 64, 192, 192), 20, (34, 40, 52, 255))
d.rounded_rectangle((64, 64, 192, 192), radius=20, outline=(120, 140, 170, 160), width=2)

# 芯片引脚
pin = (150, 165, 185, 255)
for i in range(4):
    y = 84 + i * 28
    d.rectangle((40, y, 62, y + 12), fill=pin)
    d.rectangle((194, y, 216, y + 12), fill=pin)

# 中心高光
rr((92, 92, 164, 164), 12, (52, 62, 80, 255))

# 划痕（红橙渐变斜线）
d.line((150, 60, 96, 196), fill=(235, 87, 87, 255), width=9)
d.line((150, 60, 96, 196), fill=(255, 140, 105, 160), width=4)

# 小警示点
d.ellipse((186, 178, 214, 206), fill=(242, 201, 76, 255))
d.ellipse((192, 184, 208, 200), fill=(20, 24, 32, 255))
d.rectangle((198, 187, 202, 196), fill=(242, 201, 76, 255))
d.rectangle((198, 197, 202, 200), fill=(242, 201, 76, 255))

sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
img.save(os.path.join(OUT, "app.ico"), sizes=sizes)
img.save(os.path.join(OUT, "app.png"))
print("icon written:", os.path.join(OUT, "app.ico"))
