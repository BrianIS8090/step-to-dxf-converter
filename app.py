#!/usr/bin/env python3
"""
STEP to DXF Converter - Optimized
Генерирует DXF чертежи из STEP файлов с silhouette edges
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import ezdxf
import trimesh
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
import re
import subprocess
import tempfile
import matplotlib
matplotlib.use('Agg')  # Backend without GUI
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

app = FastAPI(title="STEP to DXF Converter")

# Directory for temporary files
TEMP_DIR = Path("/tmp/step_converter")
TEMP_DIR.mkdir(exist_ok=True)

# Static files for frontend
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


def count_parts_by_name(scene):
    """
    Подсчитывает количество деталей с одинаковыми именами
    Возвращает словарь {имя: количество}
    """
    from collections import Counter
    import re
    
    if not isinstance(scene, trimesh.Scene):
        return {"Деталь": 1}
    
    names = []
    for name in scene.geometry.keys():
        # Убираем суффиксы _1, _2, _3 и пробелы в конце
        clean_name = re.sub(r'\s*_\d+\s*$', '', name).strip()
        if not clean_name:
            clean_name = "Деталь"
        names.append(clean_name)
    
    return dict(Counter(names))


def load_step_and_generate_dxf(step_path: str, output_dxf: str):
    """
    Загружает STEP и генерирует DXF с silhouette edges
    """
    # Загружаем
    scene = trimesh.load(step_path)
    
    # Глобальные bounds
    if isinstance(scene, trimesh.Scene):
        all_bounds = []
        for name, mesh in scene.geometry.items():
            all_bounds.append(mesh.bounds)
        
        if not all_bounds:
            return None
            
        all_bounds = np.array(all_bounds)
        global_bounds = np.array([
            all_bounds[:, 0].min(axis=0),
            all_bounds[:, 1].max(axis=0)
        ])
        parts_count = len(scene.geometry)
        parts_list = count_parts_by_name(scene)
    else:
        global_bounds = scene.bounds
        scene = trimesh.Scene(scene)
        parts_count = 1
        parts_list = {"Деталь": 1}
    
    # Размеры в мм
    w = (global_bounds[1][0] - global_bounds[0][0]) * 1000
    h = (global_bounds[1][1] - global_bounds[0][1]) * 1000
    d = (global_bounds[1][2] - global_bounds[0][2]) * 1000
    
    # Масштаб
    scale = min(120 / max(w, h, d), 1.0)
    
    # DXF
    doc = ezdxf.new()
    msp = doc.modelspace()
    
    min_x, min_y, min_z = global_bounds[0]
    max_x, max_y, max_z = global_bounds[1]
    
    # Размеры в DXF единицах (с учётом масштаба)
    w_pts = w * scale
    h_pts = h * scale
    d_pts = d * scale
    
    # Позиции видов с достаточными отступами
    gap = 50  # Отступ между видами
    margin = 80  # Отступ от краёв для размерных линий
    spec_width = 160  # Ширина спецификации
    
    # Вид спереди (XZ) - слева снизу
    front_x, front_y = margin + 30, margin + 30
    
    # Вид сверху (XY) - над видом спереди
    top_x, top_y = front_x, front_y + d_pts + gap + margin

    # Изометрический вид - справа от вида сверху
    # Размер изометрии примерно = диагональ bounding box * scale
    iso_size = max(w_pts, h_pts, d_pts) * 1.4  # Больше для изометрии (учитываем проекцию)
    iso_x, iso_y = front_x + w_pts + gap + margin, top_y

    # Вид сбоку (YZ) - ПОД изометрией
    side_x, side_y = iso_x, front_y
    
    # Спецификация - справа от всех проекций
    spec_x = side_x + h_pts + gap + margin
    spec_y = top_y + max(h_pts, d_pts, iso_size) - 20  # Верхний край по самому высокому виду
    
    total_lines = 0
    
    # Объединяем все детали в один mesh для изометрии (чтобы не просвечивали внутренние детали)
    all_meshes = list(scene.geometry.values())
    combined_mesh = trimesh.util.concatenate(all_meshes) if len(all_meshes) > 1 else all_meshes[0]
    
    # Обрабатываем каждую деталь для ортогональных видов
    for name, mesh in scene.geometry.items():
        # Вид сверху (XY)
        n1 = project_silhouette(msp, mesh, (0, 0, 1), top_x, top_y, scale, (min_x, min_y))
        # Вид спереди (XZ)
        n2 = project_silhouette(msp, mesh, (0, 1, 0), front_x, front_y, scale, (min_x, min_z))
        # Вид сбоку (YZ)
        n4 = project_silhouette(msp, mesh, (1, 0, 0), side_x, side_y, scale, (min_y, min_z))
        total_lines += n1 + n2 + n4
    
    # Изометрический вид - используем HLR через OCCT (без просвечивания!)
    n3 = project_isometric_hlr(step_path, msp, iso_x, iso_y, scale)
    total_lines += n3
    
    # Подписи
    msp.add_text("Вид спереди", dxfattribs={"height": 4}).set_placement((front_x, front_y - 15))
    msp.add_text("Вид сверху", dxfattribs={"height": 4}).set_placement((top_x, top_y - 15))
    msp.add_text("Изометрия", dxfattribs={"height": 4}).set_placement((iso_x, iso_y - 15))
    msp.add_text("Вид сбоку", dxfattribs={"height": 4}).set_placement((side_x, side_y - 15))
    
    # Размерные линии
    # Вид спереди - ширина и глубина
    add_dimension_line(msp, front_x, front_y, front_x + w * scale, front_y, f"{w:.0f}", offset=20)
    add_dimension_line(msp, front_x, front_y, front_x, front_y + d * scale, f"{d:.0f}", offset=25)
    
    # Вид сверху - ширина и высота
    add_dimension_line(msp, top_x, top_y, top_x + w * scale, top_y, f"{w:.0f}", offset=20)
    add_dimension_line(msp, top_x, top_y, top_x, top_y + h * scale, f"{h:.0f}", offset=25)
    
    # Вид сбоку - высота и глубина
    add_dimension_line(msp, side_x, front_y, side_x + h * scale, front_y, f"{h:.0f}", offset=20)
    add_dimension_line(msp, side_x, front_y, side_x, front_y + d * scale, f"{d:.0f}", offset=25)
    
    # Штамп
    stamp_x, stamp_y = 50, 10
    msp.add_lwpolyline([
        (stamp_x, stamp_y), (stamp_x + 200, stamp_y),
        (stamp_x + 200, stamp_y + 50), (stamp_x, stamp_y + 50), (stamp_x, stamp_y)
    ])
    msp.add_text("STEP to DXF Converter", dxfattribs={"height": 6}).set_placement((stamp_x + 5, stamp_y + 38))
    msp.add_text(f"Габариты: {w:.0f} x {h:.0f} x {d:.0f} мм", dxfattribs={"height": 3.5}).set_placement((stamp_x + 5, stamp_y + 27))
    msp.add_text(f"Деталей: {parts_count} | Масштаб: 1:{1/scale:.4f}", dxfattribs={"height": 3}).set_placement((stamp_x + 5, stamp_y + 18))
    msp.add_text(f"Линий: {total_lines}", dxfattribs={"height": 2.5}).set_placement((stamp_x + 5, stamp_y + 10))
    
    # Спецификация (список деталей)
    if parts_list and len(parts_list) > 0:
        # Позиция спецификации - справа от проекций
        line_height = 7
        
        # Заголовок спецификации
        msp.add_lwpolyline([
            (spec_x, spec_y), (spec_x + spec_width, spec_y),
            (spec_x + spec_width, spec_y + line_height * 2), (spec_x, spec_y + line_height * 2), (spec_x, spec_y)
        ])
        msp.add_text("СПЕЦИФИКАЦИЯ", dxfattribs={"height": 4}).set_placement((spec_x + 5, spec_y + 5))
        
        # Список деталей
        current_y = spec_y - line_height
        for part_name, count in sorted(parts_list.items()):
            # Обрезаем длинные имена
            display_name = part_name[:30] + "..." if len(part_name) > 30 else part_name
            
            # Имя детали
            msp.add_text(display_name, dxfattribs={"height": 2.8}).set_placement((spec_x + 5, current_y))
            
            # Количество
            msp.add_text(f"{count} шт.", dxfattribs={"height": 2.8}).set_placement((spec_x + spec_width - 35, current_y))
            
            # Линия разделитель
            msp.add_line((spec_x + 3, current_y - 2), (spec_x + spec_width - 3, current_y - 2))
            
            current_y -= line_height
            
            # Ограничиваем высоту списка
            if current_y < front_y - margin:
                msp.add_text("...", dxfattribs={"height": 2.5}).set_placement((spec_x + 5, current_y + 3))
                break
    
    doc.saveas(output_dxf)
    
    return {
        "width": w,
        "height": h,
        "depth": d,
        "parts": parts_count,
        "lines": total_lines,
        "scale": 1/scale
    }


def add_dimension_line(msp, x1, y1, x2, y2, text, offset=10, text_offset=5):
    """
    Рисует размерную линию с засечками и текстом
    """
    # Основная линия
    if x1 == x2:  # Вертикальная
        # Линия выноски
        msp.add_line((x1 - offset, y1), (x1 - offset, y2))
        # Засечки
        msp.add_line((x1 - offset - 3, y1), (x1 - offset + 3, y1))
        msp.add_line((x1 - offset - 3, y2), (x1 - offset + 3, y2))
        # Текст
        mid_y = (y1 + y2) / 2
        msp.add_text(text, dxfattribs={"height": 3, "rotation": 90}).set_placement((x1 - offset - text_offset, mid_y))
    else:  # Горизонтальная
        # Линия выноски
        msp.add_line((x1, y1 - offset), (x2, y2 - offset))
        # Засечки
        msp.add_line((x1, y1 - offset - 3), (x1, y1 - offset + 3))
        msp.add_line((x2, y2 - offset - 3), (x2, y2 - offset + 3))
        # Текст
        mid_x = (x1 + x2) / 2
        msp.add_text(text, dxfattribs={"height": 3}).set_placement((mid_x, y1 - offset - text_offset))


def project_isometric_hlr(step_path, msp, offset_x, offset_y, scale):
    """
    Генерирует изометрию с HLR через OCCT
    Вызывает внешний скрипт hlr_isometric.py
    """
    try:
        # Временный файл для HLR результата
        with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp:
            tmp_dxf = tmp.name
        
        # Вызываем HLR скрипт
        hlr_script = Path(__file__).parent / "hlr_isometric.py"
        result = subprocess.run(
            [str(hlr_script), step_path, tmp_dxf],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        if result.returncode != 0:
            print(f"HLR failed: {result.stderr}")
            return 0
        
        # Читаем результат DXF
        hlr_doc = ezdxf.readfile(tmp_dxf)
        hlr_msp = hlr_doc.modelspace()
        
        # Копируем линии в основной DXF со смещением
        lines_count = 0
        for entity in hlr_msp:
            if entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                
                # Применяем масштаб и смещение
                msp.add_line(
                    (start[0] * scale + offset_x, start[1] * scale + offset_y),
                    (end[0] * scale + offset_x, end[1] * scale + offset_y)
                )
                lines_count += 1
        
        # Удаляем временный файл
        Path(tmp_dxf).unlink(missing_ok=True)
        
        return lines_count
        
    except Exception as e:
        print(f"HLR error: {e}")
        return 0


def project_isometric(msp, mesh, offset_x, offset_y, scale, bounds):
    """
    Проецирует силуэт в изометрии (30° угол)
    Показывает контуры каждой детали (может просвечиваться)
    """
    import math
    
    faces = mesh.faces
    vertices = mesh.vertices
    face_normals = mesh.face_normals
    
    # Изометрический угол (30°)
    angle = math.radians(30)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    # Направление взгляда для изометрии
    view_dir = np.array([sin_a, sin_a, cos_a])
    view_dir = view_dir / np.linalg.norm(view_dir)
    
    # Определяем видимые грани
    visible = np.dot(face_normals, view_dir) > 0
    
    # Собираем edges
    edges_dict = defaultdict(list)
    for face_idx, face in enumerate(faces):
        for i in range(3):
            e = tuple(sorted([face[i], face[(i+1) % 3]]))
            edges_dict[e].append(face_idx)
    
    # Находим silhouette edges для изометрии
    silhouette_edges = []
    for edge, face_indices in edges_dict.items():
        if len(face_indices) == 2:
            # Ребро между двумя гранями
            if visible[face_indices[0]] != visible[face_indices[1]]:
                silhouette_edges.append(edge)
        elif len(face_indices) == 1 and visible[face_indices[0]]:
            # Boundary edge
            silhouette_edges.append(edge)
    
    # Проекция вершин в изометрию
    def project_vertex_iso(v):
        # Изометрическая проекция
        x = (v[0] - v[1]) * cos_a
        y = v[2] * cos_a + (v[0] + v[1]) * sin_a
        return x, y
    
    # Рисуем силуэт
    for e in silhouette_edges:
        v1, v2 = vertices[e[0]], vertices[e[1]]
        
        # Сдвигаем к началу координат
        v1_shifted = v1 - bounds[0]
        v2_shifted = v2 - bounds[0]
        
        # Проецируем в изометрию
        p1 = project_vertex_iso(v1_shifted)
        p2 = project_vertex_iso(v2_shifted)
        
        # Масштаб и смещение
        x1 = p1[0] * scale * 1000 + offset_x
        y1 = p1[1] * scale * 1000 + offset_y
        x2 = p2[0] * scale * 1000 + offset_x
        y2 = p2[1] * scale * 1000 + offset_y
        
        msp.add_line((x1, y1), (x2, y2))
    
    return len(silhouette_edges)


def add_dimension_line(msp, x1, y1, x2, y2, text, offset=15):
    """
    Добавляет размерную линию с засечками и текстом
    """
    if abs(x2 - x1) > abs(y2 - y1):  # Горизонтальная
        y = min(y1, y2) - offset
        msp.add_line((x1, y), (x2, y))
        msp.add_line((x1, y - 4), (x1, y + 4))
        msp.add_line((x2, y - 4), (x2, y + 4))
        msp.add_text(text, dxfattribs={"height": 3}).set_placement(((x1 + x2) / 2, y - 8))
    else:  # Вертикальная
        x = min(x1, x2) - offset
        msp.add_line((x, y1), (x, y2))
        msp.add_line((x - 4, y1), (x + 4, y1))
        msp.add_line((x - 4, y2), (x + 4, y2))
        msp.add_text(text, dxfattribs={"height": 3, "rotation": 90}).set_placement((x - 8, (y1 + y2) / 2))


def project_silhouette(msp, mesh, view_dir, global_offset_x, global_offset_y, scale, bounds_offset):
    """
    Проецирует silhouette edges на плоскость
    """
    faces = mesh.faces
    vertices = mesh.vertices
    face_normals = mesh.face_normals
    
    view = np.array(view_dir)
    visible = np.dot(face_normals, view) > 0
    
    # Собираем edges
    edges_dict = defaultdict(list)
    for face_idx, face in enumerate(faces):
        for i in range(3):
            e = tuple(sorted([face[i], face[(i+1) % 3]]))
            edges_dict[e].append(face_idx)
    
    # Находим silhouette edges
    silhouette_edges = []
    for edge, face_indices in edges_dict.items():
        if len(face_indices) == 2:
            if visible[face_indices[0]] != visible[face_indices[1]]:
                silhouette_edges.append(edge)
        elif len(face_indices) == 1 and visible[face_indices[0]]:
            silhouette_edges.append(edge)
    
    # Рисуем
    count = 0
    for e in silhouette_edges:
        v1, v2 = vertices[e[0]], vertices[e[1]]
        
        if view_dir == (0, 0, 1):  # XY
            p1 = (v1[0], v1[1])
            p2 = (v2[0], v2[1])
        elif view_dir == (0, 1, 0):  # XZ
            p1 = (v1[0], v1[2])
            p2 = (v2[0], v2[2])
        else:  # YZ
            p1 = (v1[1], v1[2])
            p2 = (v2[1], v2[2])
        
        x1 = (p1[0] - bounds_offset[0]) * scale * 1000 + global_offset_x
        y1 = (p1[1] - bounds_offset[1]) * scale * 1000 + global_offset_y
        x2 = (p2[0] - bounds_offset[0]) * scale * 1000 + global_offset_x
        y2 = (p2[1] - bounds_offset[1]) * scale * 1000 + global_offset_y
        
        msp.add_line((x1, y1), (x2, y2))
        count += 1
    
    return count


@app.post("/convert-pdf")
async def convert_to_pdf(file: UploadFile = File(...)):
    """
    Конвертирует STEP файл в PDF с чертежом
    """
    try:
        # Сохраняем загруженный файл
        step_path = TEMP_DIR / f"{file.filename}"
        with open(step_path, "wb") as f:
            f.write(await file.read())
        
        # Генерируем DXF
        dxf_path = TEMP_DIR / f"{file.filename}.dxf"
        result = load_step_and_generate_dxf(str(step_path), str(dxf_path))
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to generate DXF")
        
        # Конвертируем DXF в PDF
        pdf_path = TEMP_DIR / f"{file.filename}.pdf"
        dxf_to_pdf(str(dxf_path), str(pdf_path))
        
        # Возвращаем PDF
        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename=f"{file.filename}.pdf"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def dxf_to_pdf(dxf_path: str, pdf_path: str):
    """
    Конвертирует DXF файл в PDF через matplotlib
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    
    # Создаём figure с большим размером для качества
    fig = plt.figure(figsize=(24, 18), dpi=150)
    ax = fig.add_subplot()
    
    # Рендерим DXF через ezdxf
    ctx = RenderContext(doc)
    out = MatplotlibBackend(ax)
    frontend = Frontend(ctx, out)
    frontend.draw_layout(msp, finalize=True)
    
    # Сохраняем в PDF
    plt.tight_layout()
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)


@app.get("/", response_class=HTMLResponse)
async def index():
    """Главная страница"""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        content = html_path.read_text(encoding="utf-8")
        return HTMLResponse(
            content=content,
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"}
        )
    return HTMLResponse(content="<h1>STEP to DXF Converter</h1><p>Frontend not found</p>")


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "ok", "service": "step-to-dxf-converter"}


@app.post("/preview")
async def preview_step(file: UploadFile = File(...)):
    """
    Возвращает информацию о STEP файле
    """
    if not file.filename.lower().endswith((".step", ".stp")):
        raise HTTPException(status_code=400, detail="Только STEP/STP файлы")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_step = TEMP_DIR / f"preview_{timestamp}.step"
    
    try:
        content = await file.read()
        temp_step.write_bytes(content)
        
        # Быстрая загрузка для получения размеров
        scene = trimesh.load(str(temp_step))
        
        if isinstance(scene, trimesh.Scene):
            all_bounds = []
            for name, mesh in scene.geometry.items():
                all_bounds.append(mesh.bounds)
            
            all_bounds = np.array(all_bounds)
            bounds = np.array([all_bounds[:, 0].min(axis=0), all_bounds[:, 1].max(axis=0)])
            parts = len(scene.geometry)
        else:
            bounds = scene.bounds
            parts = 1
        
        dimensions = {
            "width": float(bounds[1][0] - bounds[0][0]) * 1000,
            "height": float(bounds[1][1] - bounds[0][1]) * 1000,
            "depth": float(bounds[1][2] - bounds[0][2]) * 1000,
            "parts": parts
        }
        
        temp_step.unlink()
        
        return {"dimensions": dimensions, "filename": file.filename}
        
    except Exception as e:
        if temp_step.exists():
            temp_step.unlink()
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@app.post("/convert")
async def convert_step_to_dxf(file: UploadFile = File(...)):
    """
    Конвертирует STEP в DXF
    """
    if not file.filename.lower().endswith((".step", ".stp")):
        raise HTTPException(status_code=400, detail="Только STEP/STP файлы")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_step = TEMP_DIR / f"input_{timestamp}.step"
    temp_dxf = TEMP_DIR / f"output_{timestamp}.dxf"
    
    try:
        content = await file.read()
        temp_step.write_bytes(content)
        
        result = load_step_and_generate_dxf(str(temp_step), str(temp_dxf))
        
        if result is None:
            raise HTTPException(status_code=500, detail="Не удалось обработать STEP файл")
        
        temp_step.unlink()
        
        return FileResponse(
            path=str(temp_dxf),
            media_type="application/dxf",
            filename=f"{Path(file.filename).stem}_drawing.dxf"
        )
        
    except Exception as e:
        if temp_step.exists():
            temp_step.unlink()
        if temp_dxf.exists():
            temp_dxf.unlink()
        raise HTTPException(status_code=500, detail=f"Ошибка конвертации: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
