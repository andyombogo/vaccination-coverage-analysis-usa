from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"

BACKGROUND = "#F8FBFC"
CARD = "#FFFFFF"
BORDER = "#D4E3E8"
TEXT = "#0F3948"
SUBTLE = "#5E7480"
ACCENT = "#0F6D80"
ACCENT_LIGHT = "#78B7EA"
ACCENT_GOLD = "#D5A52B"
ACCENT_GREEN = "#4F7C57"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                r"C:\Windows\Fonts\georgiab.ttf",
                r"C:\Windows\Fonts\arialbd.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                r"C:\Windows\Fonts\georgia.ttf",
                r"C:\Windows\Fonts\arial.ttf",
            ]
        )

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def save(image: Image.Image, name: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    image.save(ASSETS / name, format="PNG", optimize=True)


def gradient_rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], left: str, right: str) -> None:
    x0, y0, x1, y1 = box
    left_rgb = ImageColor.getrgb(left)
    right_rgb = ImageColor.getrgb(right)
    width = max(x1 - x0, 1)
    for offset in range(width):
        mix = offset / width
        color = tuple(
            int((1 - mix) * left_rgb[index] + mix * right_rgb[index]) for index in range(3)
        )
        draw.line([(x0 + offset, y0), (x0 + offset, y1)], fill=color)


def rounded_panel(image: Image.Image, box: tuple[int, int, int, int], fill: str, outline: str) -> None:
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(box, radius=24, fill=fill, outline=outline, width=2)


def make_hero() -> Image.Image:
    image = Image.new("RGB", (1200, 360), BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle((8, 16, 1192, 194), radius=24, fill="#EEF6F8", outline="#C7DADF", width=2)
    for x in range(8, 1192):
        mix = (x - 8) / 1184
        color = (
            int((1 - mix) * 238 + mix * 255),
            int((1 - mix) * 246 + mix * 248),
            int((1 - mix) * 248 + mix * 224),
        )
        draw.line([(x, 16), (x, 194)], fill=color)

    draw.text((40, 46), "US Maternal Vaccination Coverage Dashboard", fill=TEXT, font=font(34, bold=True))
    draw.text(
        (40, 114),
        "Explore CDC vaccination coverage estimates for pregnant women across vaccines, geographies, seasons, and demographic",
        fill=TEXT,
        font=font(20),
    )
    draw.text(
        (40, 148),
        "dimensions. This dashboard is optimized for lightweight cloud deployment and quick policy-oriented exploration.",
        fill=TEXT,
        font=font(20),
    )

    draw.text(
        (8, 220),
        "4,322 cleaned records loaded from the CDC pregnancy vaccination coverage dataset.",
        fill=SUBTLE,
        font=font(15),
    )
    draw.text((8, 260), "Records", fill=TEXT, font=font(18))
    draw.text((8, 306), "4,322", fill=TEXT, font=font(34))
    draw.text((404, 260), "Average coverage", fill=TEXT, font=font(18))
    draw.text((404, 306), "62.6%", fill=TEXT, font=font(34))
    draw.text((808, 260), "Weighted coverage", fill=TEXT, font=font(18))
    draw.text((808, 306), "61.7%", fill=TEXT, font=font(34))
    return image


def make_trend() -> Image.Image:
    image = Image.new("RGB", (900, 500), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((18, 18, 882, 482), radius=28, fill="#FCFDFC", outline="#E4EEF1")
    draw.text((56, 54), "Coverage trend", fill=TEXT, font=font(34, bold=True))
    draw.rounded_rectangle((70, 112, 770, 416), radius=18, fill="#FFFFFF", outline="#EEF3F5")

    for y in range(140, 370, 38):
        draw.line((110, y, 744, y), fill="#D7E3E7", width=1)

    y_labels = ["80", "70", "60", "50", "40", "30", "20"]
    for index, label in enumerate(y_labels):
        draw.text((80, 130 + index * 38), label, fill=SUBTLE, font=font(16))

    years = [str(year) for year in range(2012, 2023)]
    for index, year in enumerate(years):
        draw.text((135 + index * 55, 424), year, fill="#7C8E97", font=font(14))

    draw.text((330, 466), "Survey year / influenza season", fill=SUBTLE, font=font(18))
    draw.text((110, 92), "Average coverage (%)", fill=SUBTLE, font=font(16))

    flu = [(140, 278), (195, 228), (250, 218), (305, 214), (360, 202), (415, 208), (470, 202), (525, 188), (580, 188), (635, 224), (690, 224)]
    tdap = [(140, 336), (195, 268), (250, 190), (305, 152), (360, 116), (415, 108), (470, 102), (525, 94), (580, 96), (635, 108), (690, 102)]
    draw.line(flu, fill="#0B67C1", width=4, joint="curve")
    draw.line(tdap, fill=ACCENT_LIGHT, width=4, joint="curve")
    for point in flu:
        draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill="#0B67C1")
    for point in tdap:
        draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=ACCENT_LIGHT)

    draw.text((792, 148), "Vaccine", fill=SUBTLE, font=font(18))
    draw.ellipse((800, 180, 816, 196), fill="#0B67C1")
    draw.text((826, 176), "Influenza", fill=SUBTLE, font=font(17))
    draw.ellipse((800, 214, 816, 230), fill=ACCENT_LIGHT)
    draw.text((826, 210), "Tdap", fill=SUBTLE, font=font(17))
    return image


def make_geographies() -> Image.Image:
    image = Image.new("RGB", (900, 560), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((18, 18, 882, 542), radius=28, fill="#FCFDFC", outline="#E4EEF1")

    tabs = [("Overview", 52), ("Geographies", 138), ("Equity Lens", 242), ("Data and Deployment", 346)]
    for label, x in tabs:
        draw.text((x, 44), label, fill=TEXT, font=font(16))
    draw.line((134, 66, 222, 66), fill="#0B7A9A", width=3)
    draw.line((52, 74, 848, 74), fill="#D5E4E8", width=1)

    draw.text((52, 118), "Highest-coverage geographies", fill=TEXT, font=font(34, bold=True))
    labels = [
        "Commonwealt...", "Massachusetts", "New Hampshire", "Nebraska", "Minnesota", "Iowa",
        "Vermont", "Washington", "South Dakota", "District of Colu...", "Montana", "Utah",
    ]
    bar_colors = ["#1E5B93", "#2B6FA3", "#3A86B5", "#4793BF", "#56A2C4", "#63ADC9", "#72B8CD", "#86C2D1", "#9BCCD6", "#AAD5D8", "#B4DDD8", "#BDE3D8"]
    bar_widths = [650, 610, 592, 586, 580, 578, 572, 566, 562, 558, 554, 552]
    for index, label in enumerate(labels):
        y = 172 + index * 32
        draw.text((52, y), label, fill="#7C8E97", font=font(16))
        draw.rounded_rectangle((190, y, 190 + bar_widths[index], y + 22), radius=8, fill=bar_colors[index])

    for x in range(190, 843, 60):
        draw.line((x, 168, x, 520), fill="#E6EEF1", width=1)

    draw.text((470, 520), "Average coverage (%)", fill=SUBTLE, font=font(18))
    draw.text((52, 146), "Geography", fill=SUBTLE, font=font(18))
    draw.rounded_rectangle((648, 212, 834, 274), radius=12, fill="#F8FAFB", outline="#D1DDE2")
    draw.text((668, 230), "Geography", fill=SUBTLE, font=font(16))
    draw.text((756, 230), "Massachusetts", fill=TEXT, font=font(16, bold=True))
    draw.text((668, 254), "Coverage", fill=SUBTLE, font=font(16))
    draw.text((748, 254), "79.3", fill=TEXT, font=font(16, bold=True))
    return image


def make_architecture() -> Image.Image:
    image = Image.new("RGB", (1280, 720), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((84, 60), "Deployment and Data Flow", fill=TEXT, font=font(40, bold=True))
    draw.text(
        (84, 116),
        "A lightweight Streamlit dashboard deployed on Render, backed by a local CDC CSV and optional offline Spark workflows.",
        fill=SUBTLE,
        font=font(22),
    )

    top_boxes = [
        ((84, 188, 336, 320), "GitHub Repo", ["Source code, README, Render blueprint,", "tests, and portfolio assets."]),
        ((510, 188, 762, 320), "Render", ["Builds from GitHub and runs", "the public Streamlit service."]),
        ((936, 188, 1188, 320), "Users", ["Explore the dashboard, filter", "views, and export results."]),
    ]
    bottom_boxes = [
        ((84, 430, 426, 600), "Dashboard Runtime", ["`dashboard.py` + Streamlit", "`requirements.txt` + theme config", "fast startup for portfolio sharing"]),
        ((470, 430, 812, 600), "Data Layer", ["`vaccination_data.csv`", "cleaned with pandas at app load", "confidence intervals parsed for charts"]),
        ((856, 430, 1188, 600), "Offline Analysis", ["`app.py` and `project.py`", "PySpark workflow kept for local", "experimentation and model work"]),
    ]

    for box, title, lines in top_boxes + bottom_boxes:
        draw.rounded_rectangle(box, radius=24, fill=CARD, outline=BORDER, width=2)
        draw.text((box[0] + 30, box[1] + 28), title, fill=TEXT, font=font(28, bold=True))
        for index, line in enumerate(lines):
            draw.text((box[0] + 30, box[1] + 72 + index * 28), line, fill="#35545D", font=font(20))

    draw.line((336, 254, 510, 254), fill=ACCENT, width=8)
    draw.line((762, 254, 936, 254), fill=ACCENT, width=8)
    draw.line((210, 320, 210, 430), fill=ACCENT_GOLD, width=8)
    draw.line((636, 320, 636, 430), fill=ACCENT_GOLD, width=8)
    draw.line((1022, 320, 1022, 430), fill=ACCENT_GOLD, width=8)
    return image


def main() -> None:
    save(make_hero(), "dashboard-hero-preview.png")
    save(make_trend(), "coverage-trend-preview.png")
    save(make_geographies(), "geographies-preview.png")
    save(make_architecture(), "architecture.png")


if __name__ == "__main__":
    from PIL import ImageColor

    main()
