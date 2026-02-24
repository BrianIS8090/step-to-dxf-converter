#!/opt/miniconda3/envs/occt/bin/python
"""
Hidden Line Removal для изометрического вида с использованием Open CASCADE
"""

import sys
import math
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCC.Core.HLRAlgo import HLRAlgo_Projector
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.TopoDS import topods
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_UniformAbscissa
import ezdxf


def load_step_file(step_path):
    """Загрузка STEP файла через OCCT"""
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != 1:
        raise Exception(f"Failed to read STEP file: {step_path}")
    
    reader.TransferRoots()
    shape = reader.OneShape()
    return shape


def compute_hlr_isometric(shape):
    """
    Вычисляет HLR для изометрической проекции - вид сверху справа
    Возвращает видимые и скрытые рёбра
    """
    # Поворот на 45° по часовой стрелке от (1, 1, 1)
    # Формула поворота в плоскости XY:
    # x' = x*cos(45) + y*sin(45)
    # y' = -x*sin(45) + y*cos(45)
    # z' = z
    angle = math.radians(45)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    # Поворачиваем (1, 1, 1) на 45° CW
    vx = 1.0 * cos_a + 1.0 * sin_a  # = 1.414
    vy = -1.0 * sin_a + 1.0 * cos_a  # = 0
    vz = 1.0
    
    view_dir = gp_Dir(vx, vy, vz)
    
    # Точка наблюдения (далеко от объекта)
    eye_point = gp_Pnt(vx * 1000, vy * 1000, vz * 1000)
    
    # X direction = перпендикулярен view_dir в плоскости XY
    # Для (1.414, 0, 1): перпендикуляр в XY = (0, 1, 0)
    x_dir = gp_Dir(0.0, 1.0, 0.0)
    
    # Создаём проектор с явным заданием координатных осей
    projector = HLRAlgo_Projector(gp_Ax2(eye_point, view_dir, x_dir))
    
    # Создаём HLR алгоритм
    hlr = HLRBRep_Algo()
    hlr.Add(shape)
    hlr.Projector(projector)
    hlr.Update()
    hlr.Hide()
    
    return hlr


def extract_edges(hlr, visible_only=True):
    """
    Извлекает рёбра из HLR результата
    visible_only=True -> только видимые
    visible_only=False -> все (видимые + скрытые)
    """
    edges = []
    
    # Конвертер HLR -> TopoDS_Shape
    hlr_to_shape = HLRBRep_HLRToShape(hlr)
    
    # Видимые контуры (VCompound)
    visible_compound = hlr_to_shape.VCompound()
    if visible_compound:
        explorer = TopExp_Explorer(visible_compound, TopAbs_EDGE)
        while explorer.More():
            edge = topods.Edge(explorer.Current())
            edges.append(('visible', edge))
            explorer.Next()
    
    # Скрытые контуры (если нужно)
    if not visible_only:
        hidden_compound = hlr_to_shape.HCompound()
        if hidden_compound:
            explorer = TopExp_Explorer(hidden_compound, TopAbs_EDGE)
            while explorer.More():
                edge = topods.Edge(explorer.Current())
                edges.append(('hidden', edge))
                explorer.Next()
    
    return edges


def project_3d_to_2d_iso(x, y, z):
    """
    Проецирует 3D точку на 2D плоскость изометрии
    Учитывает поворот на 45° по часовой стрелке
    """
    # Поворот координат на 45° по часовой стрелке в плоскости XY
    angle = math.radians(45)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    # Применяем поворот к координатам
    x_rot = x * cos_a + y * sin_a
    y_rot = -x * sin_a + y * cos_a
    z_rot = z
    
    # Стандартная изометрическая проекция для повёрнутых координат
    iso_angle = math.radians(45)
    cos_iso = math.cos(iso_angle)
    sin_iso = math.sin(iso_angle)
    
    # Изометрическая проекция
    px = (x_rot - y_rot) * cos_iso
    py = z_rot * cos_iso + (x_rot + y_rot) * sin_iso
    
    return px, py


def edge_to_dxf_lines(edge, edge_type='visible'):
    """
    Конвертирует OCCT edge в список линий для DXF
    Возвращает [(x1, y1, x2, y2), ...]
    """
    lines = []
    
    try:
        # Используем BRepAdaptor для работы с edge
        curve_adaptor = BRepAdaptor_Curve(edge)
        first = curve_adaptor.FirstParameter()
        last = curve_adaptor.LastParameter()
        
        # Дискретизация кривой
        num_points = 20
        step = (last - first) / num_points
        
        prev_point = None
        for i in range(num_points + 1):
            param = first + i * step
            point_3d = curve_adaptor.Value(param)
            
            # Проецируем 3D точку на 2D плоскость изометрии
            px, py = project_3d_to_2d_iso(
                point_3d.X(),
                point_3d.Y(),
                point_3d.Z()
            )
            
            if prev_point:
                lines.append((
                    prev_point[0], prev_point[1],
                    px, py
                ))
            prev_point = (px, py)
    except Exception as e:
        # Если не удалось извлечь кривую, пропускаем edge
        pass
    
    return lines


def hlr_to_dxf(step_path, output_dxf, offset_x=100, offset_y=100, scale=1.0):
    """
    Главная функция: STEP -> HLR -> DXF
    """
    print(f"Loading STEP file: {step_path}")
    shape = load_step_file(step_path)
    
    print("Computing HLR for isometric view...")
    hlr = compute_hlr_isometric(shape)
    
    print("Extracting visible edges...")
    edges = extract_edges(hlr, visible_only=True)
    
    print(f"Found {len(edges)} edges")
    
    # Создаём DXF
    doc = ezdxf.new()
    msp = doc.modelspace()
    
    # Первый проход: вычисляем bounding box
    all_lines = []
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    
    for edge_type, edge in edges:
        lines = edge_to_dxf_lines(edge, edge_type)
        for x1, y1, x2, y2 in lines:
            all_lines.append((x1, y1, x2, y2))
            min_x = min(min_x, x1, x2)
            max_x = max(max_x, x1, x2)
            min_y = min(min_y, y1, y2)
            max_y = max(max_y, y1, y2)
    
    # Нормализуем координаты: сдвигаем к (0, 0)
    total_lines = 0
    for x1, y1, x2, y2 in all_lines:
        # Сдвигаем к началу координат
        nx1 = x1 - min_x
        ny1 = y1 - min_y
        nx2 = x2 - min_x
        ny2 = y2 - min_y
        
        # Масштаб и смещение
        msp.add_line(
            (nx1 * scale + offset_x, ny1 * scale + offset_y),
            (nx2 * scale + offset_x, ny2 * scale + offset_y)
        )
        total_lines += 1
    
    doc.saveas(output_dxf)
    print(f"DXF saved: {output_dxf} ({total_lines} lines)")
    print(f"Bounding box: {max_x - min_x:.1f} x {max_y - min_y:.1f}")
    
    return total_lines


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: hlr_isometric.py <input.step> <output.dxf>")
        sys.exit(1)
    
    step_path = sys.argv[1]
    output_dxf = sys.argv[2]
    
    try:
        total_lines = hlr_to_dxf(step_path, output_dxf)
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
