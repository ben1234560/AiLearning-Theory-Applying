"""第二章原理演示：二维坐标变换、动作归一化和 SVG 可视化。"""

from pathlib import Path


def matvec(matrix, vector):
    return [sum(row[j] * vector[j] for j in range(2)) for row in matrix]


def add(left, right):
    return [left[i] + right[i] for i in range(2)]


def fmt(vector):
    return f"[{vector[0]:g}, {vector[1]:g}]"


def to_canvas(point):
    """数学坐标 y 向上；SVG 画布坐标 y 向下。"""
    return 100 + 80 * point[0], 370 - 80 * point[1]


def main():
    rotation = [[0.0, -1.0], [1.0, 0.0]]
    translation_a_b = [3.0, 1.0]
    point_b = [2.0, 1.0]
    direction_b = [1.0, 0.0]

    rotated = matvec(rotation, point_b)
    point_a = add(rotated, translation_a_b)
    direction_a = matvec(rotation, direction_b)

    action = [384.0, 128.0]
    normalized = [2 * value / 512 - 1 for value in action]
    recovered = [512 * (value + 1) / 2 for value in normalized]

    assert point_a == [2.0, 3.0]
    assert direction_a == [0.0, 1.0]
    assert normalized == [0.5, -0.5]
    assert recovered == action

    print(f"p^A = {fmt(point_a)}")
    print(f"delta p^A = {fmt(direction_a)}")
    print(f"action = {fmt(action)} -> normalized = {fmt(normalized)} -> recovered = {fmt(recovered)}")

    origin = to_canvas([0, 0])
    frame_b = to_canvas(translation_a_b)
    point = to_canvas(point_a)
    arrow_end = to_canvas(add(point_a, direction_a))

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="520" height="440" viewBox="0 0 520 440">
<rect width="520" height="440" fill="white"/>
<text x="25" y="30" font-family="sans-serif" font-size="18">Chapter 2: 2D coordinate transform</text>
<line x1="100" y1="370" x2="470" y2="370" stroke="#555" stroke-width="2"/>
<line x1="100" y1="400" x2="100" y2="50" stroke="#555" stroke-width="2"/>
<text x="475" y="375" font-family="sans-serif">xA</text>
<text x="92" y="45" font-family="sans-serif">yA</text>
<circle cx="{origin[0]}" cy="{origin[1]}" r="5" fill="#555"/>
<text x="108" y="389" font-family="sans-serif">Origin A (0,0)</text>
<circle cx="{frame_b[0]}" cy="{frame_b[1]}" r="7" fill="#e67e22"/>
<text x="{frame_b[0]+12}" y="{frame_b[1]-12}" font-family="sans-serif">Origin B (3,1)</text>
<line x1="{frame_b[0]}" y1="{frame_b[1]}" x2="{point[0]}" y2="{point[1]}" stroke="#2980b9" stroke-width="2" stroke-dasharray="6 5"/>
<circle cx="{point[0]}" cy="{point[1]}" r="8" fill="#2980b9"/>
<text x="{point[0]+12}" y="{point[1]}" font-family="sans-serif">Point pA = (2,3)</text>
<line x1="{point[0]}" y1="{point[1]}" x2="{arrow_end[0]}" y2="{arrow_end[1]}" stroke="#27ae60" stroke-width="3"/>
<polygon points="{arrow_end[0]},{arrow_end[1]-7} {arrow_end[0]-6},{arrow_end[1]+7} {arrow_end[0]+6},{arrow_end[1]+7}" fill="#27ae60"/>
<text x="{arrow_end[0]+12}" y="{arrow_end[1]}" font-family="sans-serif">Delta pA = (0,1)</text>
<text x="25" y="420" font-family="sans-serif" font-size="12" fill="#666">Geometry illustration; not a robot workspace.</text>
</svg>'''
    output = Path(__file__).with_name("chapter02_coordinates.svg")
    output.write_text(svg, encoding="utf-8")
    print(f"visualization = {output.name}")


if __name__ == "__main__":
    main()
