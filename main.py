import base64
import time
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

IMAGE_PATH = Path(__file__).parent / "photos" / "text.jpg"
RESULTS_DIR = Path(__file__).parent / "results"
BLOCK_SIZE = 25
C = 15
MAX_VALUE = 255
MAX_SIDE = 600
PREVIEW_SIDE = 420

BENCH_SIDES = [100, 200, 300, 400, 500, 600]
BENCH_BLOCK_SIZES = [3, 7, 11, 15, 21, 25, 31]
BENCH_SIDE = 300
LIBRARY_REPEATS = 20
NATIVE_REPEATS = 1

LIBRARY_NAME = "OpenCV"
NATIVE_NAME = "Собственная реализация"
LIBRARY_COLOR = "#2a78d6"
NATIVE_COLOR = "#eb6834"
SURFACE_COLOR = "#fcfcfb"
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e3e2de"


def threshold_library(gray, block_size, c):
    return cv2.adaptiveThreshold(
        gray,
        MAX_VALUE,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )


def threshold_native(gray, block_size, c):
    height, width = gray.shape
    radius = block_size // 2
    area = block_size * block_size
    rows = gray.tolist()

    padded = [[row[0]] * radius + row + [row[-1]] * radius for row in rows]
    padded = [padded[0]] * radius + padded + [padded[-1]] * radius

    result = []
    for y in range(height):
        window_rows = padded[y:y + block_size]
        source_row = rows[y]
        result_row = []
        for x in range(width):
            total = 0
            for window_row in window_rows:
                total += sum(window_row[x:x + block_size])
            mean = total // area
            result_row.append(MAX_VALUE if source_row[x] > mean - c else 0)
        result.append(result_row)

    return np.array(result, dtype=np.uint8)


def resize_to_side(image, side):
    height, width = image.shape[:2]
    scale = side / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    if max(image.shape[:2]) > MAX_SIDE:
        image = resize_to_side(image, MAX_SIDE)
    return image


def measure(function, gray, block_size, c, repeats=1):
    best = float("inf")
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function(gray, block_size, c)
        best = min(best, time.perf_counter() - start)
    return result, best


def to_photo(image):
    height, width = image.shape[:2]
    scale = PREVIEW_SIDE / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST
    preview = cv2.resize(image, size, interpolation=interpolation)
    encoded = cv2.imencode(".png", preview)[1]
    return tk.PhotoImage(data=base64.b64encode(encoded.tobytes()))


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Адаптивная бинаризация (mean)")
        self.photos = []
        self.results = {}

        toolbar = tk.Frame(root)
        toolbar.pack(fill=tk.X, padx=10, pady=10)
        self.status = tk.Label(toolbar, text="Обработка...")
        self.status.pack(side=tk.LEFT)

        panels = tk.Frame(root)
        panels.pack(padx=10, pady=(0, 10))
        self.panels = [
            self.create_panel(panels, 0, "Исходное изображение"),
            self.create_panel(panels, 1, LIBRARY_NAME, "cv"),
            self.create_panel(panels, 2, NATIVE_NAME, "native"),
        ]

        self.figure = Figure(figsize=(11, 3.2), dpi=100, facecolor=SURFACE_COLOR)
        self.size_axes, self.block_axes = self.figure.subplots(1, 2)
        self.style_axes(self.size_axes, "Время от размера изображения", "Размер изображения, тыс. пикселей")
        self.style_axes(self.block_axes, "Время от размера окна", "blockSize")
        self.figure.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.canvas.draw()

    def create_panel(self, parent, column, title, suffix=None):
        frame = tk.Frame(parent)
        frame.grid(row=0, column=column, padx=5, sticky=tk.N)
        tk.Label(frame, text=title, font=("TkDefaultFont", 14, "bold")).pack()
        image_label = tk.Label(frame)
        image_label.pack(pady=5)
        info_label = tk.Label(frame)
        info_label.pack()
        if suffix:
            tk.Button(frame, text="Сохранить", command=lambda: self.save(suffix)).pack(pady=(5, 0))
        return image_label, info_label

    def style_axes(self, axes, title, xlabel):
        axes.set_facecolor(SURFACE_COLOR)
        axes.set_title(title, fontsize=11, color=TEXT_COLOR, loc="left")
        axes.set_xlabel(xlabel, fontsize=9, color=MUTED_COLOR)
        axes.set_ylabel("Время, мс", fontsize=9, color=MUTED_COLOR)
        axes.set_yscale("log")
        axes.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8)
        axes.set_axisbelow(True)
        axes.tick_params(colors=MUTED_COLOR, labelsize=8, length=0, which="both")
        for side in ("top", "right", "left"):
            axes.spines[side].set_visible(False)
        axes.spines["bottom"].set_color(GRID_COLOR)

    def save(self, suffix):
        result = self.results.get(suffix)
        if result is None:
            return
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / f"{IMAGE_PATH.stem}-{suffix}.png"
        if cv2.imwrite(str(path), result):
            self.status.config(text=f"Сохранено: {RESULTS_DIR.name}/{path.name}")
        else:
            self.status.config(text=f"Не удалось сохранить {path.name}")

    def process(self):
        image = load_image(IMAGE_PATH)
        if image is None:
            self.status.config(text=f"Не удалось открыть файл {IMAGE_PATH.name}")
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        library_result, library_time = measure(threshold_library, gray, BLOCK_SIZE, C, LIBRARY_REPEATS)
        native_result, native_time = measure(threshold_native, gray, BLOCK_SIZE, C, NATIVE_REPEATS)
        self.results = {"cv": library_result, "native": native_result}

        height, width = gray.shape
        self.photos = [to_photo(image), to_photo(library_result), to_photo(native_result)]
        infos = [
            f"{width} x {height}",
            f"{library_time * 1000:.3f} мс",
            f"{native_time * 1000:.3f} мс",
        ]
        for (image_label, info_label), photo, info in zip(self.panels, self.photos, infos):
            image_label.config(image=photo)
            info_label.config(text=info)

        self.benchmark(gray)
        self.status.config(text=f"{IMAGE_PATH.name}, blockSize = {BLOCK_SIZE}, C = {C}")

    def benchmark(self, gray):
        cases = [(side, BLOCK_SIZE) for side in BENCH_SIDES]
        cases += [(BENCH_SIDE, block_size) for block_size in BENCH_BLOCK_SIZES]
        library_times = []
        native_times = []
        pixels = []

        for index, (side, block_size) in enumerate(cases, start=1):
            self.status.config(text=f"Измерение быстродействия: {index} / {len(cases)}")
            self.root.update()
            sample = resize_to_side(gray, side)
            library_times.append(measure(threshold_library, sample, block_size, C, LIBRARY_REPEATS)[1] * 1000)
            native_times.append(measure(threshold_native, sample, block_size, C, NATIVE_REPEATS)[1] * 1000)
            pixels.append(sample.size / 1000)

        split = len(BENCH_SIDES)
        self.plot(self.size_axes, pixels[:split], library_times[:split], native_times[:split])
        self.plot(self.block_axes, BENCH_BLOCK_SIZES, library_times[split:], native_times[split:])
        self.block_axes.set_xticks(BENCH_BLOCK_SIZES)
        self.size_axes.legend(
            loc="center right",
            fontsize=8,
            frameon=False,
            labelcolor=TEXT_COLOR,
        )
        self.figure.tight_layout()
        self.canvas.draw()

    def plot(self, axes, x, library_times, native_times):
        series = [
            (LIBRARY_NAME, LIBRARY_COLOR, library_times),
            (NATIVE_NAME, NATIVE_COLOR, native_times),
        ]
        for name, color, times in series:
            axes.plot(x, times, color=color, linewidth=2, marker="o", markersize=6, label=name)
            axes.annotate(
                f"{times[-1]:.0f} мс" if times[-1] >= 10 else f"{times[-1]:.2f} мс",
                (x[-1], times[-1]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="right",
                fontsize=8,
                color=TEXT_COLOR,
            )
        axes.set_ylim(min(library_times) / 3, max(native_times) * 4)


def main():
    root = tk.Tk()
    app = App(root)

    def run():
        try:
            app.process()
        except tk.TclError:
            pass

    root.after(100, run)
    root.mainloop()


if __name__ == "__main__":
    main()
