"""Indexed OpenGL surface rendering using the Qt modules already installed.

The triangle buffer and vertex storage are created once. Only positions and
normals change per tracked frame. CPU submission and optional asynchronous GPU
timing are separate; normal operation never calls glFinish or reads pixels.
"""

import math
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtOpenGL, QtOpenGLWidgets
from shiboken6 import VoidPtr

from .mesh import TRIANGLES, vertex_normals
from .core import FrameRate
from .render_common import OrbitControls

# Core OpenGL constants; functions and resource ownership come from Qt.
GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT = 0x4000, 0x0100
GL_DEPTH_TEST, GL_CULL_FACE, GL_LEQUAL = 0x0B71, 0x0B44, 0x0203
GL_FLOAT, GL_UNSIGNED_INT, GL_TRIANGLES, GL_LINES, GL_POINTS = 0x1406, 0x1405, 4, 1, 0
GL_POLYGON_OFFSET_FILL, GL_PROGRAM_POINT_SIZE = 0x8037, 0x8642
GL_RENDERER, GL_VERSION = 0x1F01, 0x1F02
INDEX_OFFSET_ZERO = VoidPtr(0)

VERTEX_SHADER = """#version 150 core
in vec3 position;
in vec3 normal;
uniform mat4 projection;
uniform mat4 model;
out vec3 viewNormal;
out vec3 viewPosition;
void main() {
    vec4 p = model * vec4(position, 1.0);
    viewPosition = p.xyz;
    viewNormal = mat3(model) * normal;
    gl_Position = projection * p;
    gl_PointSize = 2.5;
}
"""

FRAGMENT_SHADER = """#version 150 core
in vec3 viewNormal;
in vec3 viewPosition;
uniform vec3 lightDirection;
uniform float lightStrength;
uniform bool wire;
out vec4 color;
void main() {
    if (wire) { color = vec4(0.22, 0.67, 0.66, 1.0); return; }
    vec3 n = normalize(viewNormal);
    if (!gl_FrontFacing) n = -n;
    vec3 l = normalize(lightDirection);
    float diffuse = max(dot(n, l), 0.0);
    vec3 halfVector = normalize(l + vec3(0.0, 0.0, 1.0));
    float specular = pow(max(dot(n, halfVector), 0.0), 36.0) * 0.12;
    vec3 material = vec3(0.67, 0.69, 0.71);
    vec3 lit = material * (0.24 + lightStrength * 0.72 * diffuse) + lightStrength * specular;
    if (!gl_FrontFacing) lit *= 0.68;
    color = vec4(clamp(lit, 0.0, 1.0), 1.0);
}
"""


def configure_surface_format():
    """Must run before QApplication on macOS to share core-profile contexts."""
    fmt = QtGui.QSurfaceFormat()
    fmt.setRenderableType(QtGui.QSurfaceFormat.RenderableType.OpenGL)
    fmt.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setVersion(4, 1)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    fmt.setSwapInterval(1)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)


class SurfaceCanvas(OrbitControls, QtOpenGLWidgets.QOpenGLWidget):
    failed = QtCore.Signal(str)
    ready = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        fmt = QtGui.QSurfaceFormat.defaultFormat()
        fmt.setVersion(4, 1)
        fmt.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setDepthBufferSize(24)
        self.setFormat(fmt)
        self.setMinimumSize(320, 320)
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.points = None
        self.edges = np.empty((0, 2), dtype=np.uint32)
        self.reference_span = None
        self.yaw = self.pitch = 0.0
        self.zoom = 1.0
        self._drag_position = None
        self.mode = "Solid"
        self.light_angle, self.light_strength = -35.0, 1.0
        self.cpu_draw_ms = self.normal_ms = 0.0
        self.gpu_draw_ms = None
        self.renderer_info = "Initializing OpenGL…"
        self.render_error = None
        self._vertices = np.zeros((478, 6), dtype=np.float32)
        self._vertices[:, 5] = 1
        self._dirty = self._edges_dirty = False
        self._initialized = False
        self._queries = []
        self._pending_queries = []
        self._available_queries = []
        self._draw_count = 0
        self._active_query = None
        self.draw_rate = FrameRate()
        self.render_fps = 0.0
        self.frameSwapped.connect(self._frame_swapped)

    def _frame_swapped(self):
        if self.points is not None:
            self.render_fps = self.draw_rate.update(time.perf_counter())

    def update_face(self, points, edges):
        if points is None:
            self.points = None
            self.draw_rate = FrameRate()
            self.render_fps = 0.0
            self.edges = np.empty((0, 2), dtype=np.uint32)
            self.update()
            return
        points = np.asarray(points, dtype=np.float32)
        edges = np.asarray(edges)
        if (points.ndim != 2 or points.shape[1] != 3 or not 0 < len(points) <= 478
                or not np.isfinite(points).all()):
            raise ValueError("3D face points must be a finite N by 3 array, N <= 478")
        if edges.size == 0:
            edges = np.empty((0, 2), dtype=np.uint32)
        if (edges.ndim != 2 or edges.shape[1] != 2 or len(edges) > 2048 or not np.issubdtype(edges.dtype, np.integer)
                or (edges.size and (edges.min() < 0 or edges.max() >= len(points)))):
            raise ValueError("Invalid mesh connections")
        if self.points is None:
            self.draw_rate = FrameRate()
            self.render_fps = 0.0
        self.points = points.copy()
        if not np.array_equal(edges, self.edges):
            self.edges = np.ascontiguousarray(edges, dtype=np.uint32)
            self._edges_dirty = True
        count = len(points)
        self._vertices[:count, :3] = points
        start = time.perf_counter()
        if count >= 468:
            self._vertices[:count, 3:] = vertex_normals(points)
        else:
            self._vertices[:count, 3:] = (0, 0, 1)
        self.normal_ms = (time.perf_counter() - start) * 1000
        self._dirty = True
        if self.reference_span is None:
            self.reference_span = max(float(np.linalg.norm(points, axis=1).max()) * 2, 1e-3)
        self.update()

    def set_mode(self, mode):
        if mode not in ("Wireframe", "Solid", "Solid + Wireframe"):
            raise ValueError("Unknown display mode")
        self.mode = mode
        self.update()

    def set_light(self, angle, strength):
        self.light_angle, self.light_strength = float(angle), float(strength)
        self.update()

    def initializeGL(self):
        try:
            self.gl = self.context().functions()
            self.gl.initializeOpenGLFunctions()
            self.renderer_info = str(self.gl.glGetString(GL_RENDERER))
            self.program = QtOpenGL.QOpenGLShaderProgram(self)
            for kind, source in ((QtOpenGL.QOpenGLShader.ShaderTypeBit.Vertex, VERTEX_SHADER),
                                 (QtOpenGL.QOpenGLShader.ShaderTypeBit.Fragment, FRAGMENT_SHADER)):
                if not self.program.addShaderFromSourceCode(kind, source):
                    raise RuntimeError(self.program.log())
            self.program.bindAttributeLocation("position", 0)
            self.program.bindAttributeLocation("normal", 1)
            if not self.program.link():
                raise RuntimeError(self.program.log())
            self._light_strength_location = self.program.uniformLocation("lightStrength")
            self.vao = QtOpenGL.QOpenGLVertexArrayObject(self)
            if not self.vao.create():
                raise RuntimeError("Cannot create vertex array")
            self.vao.bind()
            self.vbo = QtOpenGL.QOpenGLBuffer()
            self.vbo.setUsagePattern(QtOpenGL.QOpenGLBuffer.UsagePattern.DynamicDraw)
            self.vbo.create()
            self.vbo.bind()
            self.vbo.allocate(self._vertices.nbytes)
            self.program.bind()
            self.program.enableAttributeArray(0)
            self.program.enableAttributeArray(1)
            self.program.setAttributeBuffer(0, GL_FLOAT, 0, 3, 24)
            self.program.setAttributeBuffer(1, GL_FLOAT, 12, 3, 24)
            self.triangle_buffer = self._index_buffer(TRIANGLES)
            self.edge_buffer = self._index_buffer(np.zeros((2048, 2), dtype=np.uint32), dynamic=True)
            self.vao.release()
            self.program.release()
            for _ in range(3):
                query = QtOpenGL.QOpenGLTimerQuery(self)
                if query.create():
                    self._queries.append(query)
                    self._available_queries.append(query)
                else:
                    break
            self.context().aboutToBeDestroyed.connect(self.cleanup)
            self._initialized = True
            self.ready.emit(self.renderer_info)
        except Exception as exc:
            self.render_error = str(exc)
            self.failed.emit(self.render_error)

    def _index_buffer(self, values, dynamic=False):
        buffer = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.IndexBuffer)
        buffer.setUsagePattern(QtOpenGL.QOpenGLBuffer.UsagePattern.DynamicDraw if dynamic
                               else QtOpenGL.QOpenGLBuffer.UsagePattern.StaticDraw)
        if not buffer.create() or not buffer.bind():
            raise RuntimeError("Cannot create index buffer")
        buffer.allocate(values.tobytes(), values.nbytes)
        return buffer

    def paintGL(self):
        if not self._initialized or self.render_error:
            return
        try:
            self._paint_gl()
        except Exception as exc:
            if self._active_query is not None:
                self._active_query.end()
                self._active_query = None
            self.render_error = str(exc)
            self.failed.emit(self.render_error)

    def _paint_gl(self):
        start = time.perf_counter()
        gl = self.gl
        gl.glClearColor(0.063, 0.098, 0.137, 1.0)
        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self.points is not None:
            while self._pending_queries and self._pending_queries[0].isResultAvailable():
                query = self._pending_queries.pop(0)
                self.gpu_draw_ms = query.waitForResult() / 1_000_000
                self._available_queries.append(query)
            query = self._available_queries.pop() if self._available_queries else None
            if query:
                query.begin()
                self._active_query = query
            self.vao.bind()
            self.vbo.bind()
            if self._dirty:
                self.vbo.write(0, self._vertices.data, self._vertices.nbytes)
                self._dirty = False
            if self._edges_dirty:
                self.edge_buffer.bind()
                self.edge_buffer.write(0, self.edges.data, self.edges.nbytes)
                self._edges_dirty = False
            self.program.bind()
            # Equal world-unit scale on x/y; the window aspect changes only the view.
            span = self.reference_span / (0.78 * self.zoom)
            aspect = self.width() / max(self.height(), 1)
            half_x, half_y = span / 2 * max(1, aspect), span / 2 * max(1, 1 / aspect)
            projection = QtGui.QMatrix4x4()
            projection.ortho(-half_x, half_x, -half_y, half_y, -10.0, 10.0)
            model = QtGui.QMatrix4x4()
            model.rotate(self.pitch, 1, 0, 0)
            model.rotate(self.yaw, 0, 1, 0)
            angle = math.radians(self.light_angle)
            self.program.setUniformValue("projection", projection)
            self.program.setUniformValue("model", model)
            self.program.setUniformValue("lightDirection", QtGui.QVector3D(math.sin(angle), 0.45, math.cos(angle)))
            # Explicit float upload avoids Qt Python's int/float overload ambiguity.
            gl.glUniform1f(self._light_strength_location, self.light_strength)
            gl.glEnable(GL_DEPTH_TEST)
            gl.glDepthFunc(GL_LEQUAL)
            gl.glDisable(GL_CULL_FACE)
            if self.mode != "Wireframe" and len(self.points) >= 468:
                self.program.setUniformValue("wire", False)
                self.triangle_buffer.bind()
                gl.glEnable(GL_POLYGON_OFFSET_FILL)
                gl.glPolygonOffset(1.0, 1.0)
                gl.glDrawElements(GL_TRIANGLES, int(TRIANGLES.size), GL_UNSIGNED_INT, INDEX_OFFSET_ZERO)
                gl.glDisable(GL_POLYGON_OFFSET_FILL)
            if self.mode != "Solid" or len(self.points) < 468:
                self.program.setUniformValue("wire", True)
                self.edge_buffer.bind()
                gl.glDrawElements(GL_LINES, int(self.edges.size), GL_UNSIGNED_INT, INDEX_OFFSET_ZERO)
                if self.mode == "Wireframe":
                    gl.glEnable(GL_PROGRAM_POINT_SIZE)
                    gl.glDrawArrays(GL_POINTS, 0, len(self.points))
            self.vao.release()
            self.program.release()
            if query:
                query.end()
                self._active_query = None
                self._pending_queries.append(query)
            self._draw_count += 1
        self.cpu_draw_ms = (time.perf_counter() - start) * 1000
        if self.points is None:
            painter = QtGui.QPainter(self)
            painter.setPen(QtGui.QColor("#c3ceda"))
            painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter,
                             "No face detected\nLook toward the camera to show the live surface.")

    def cleanup(self):
        if not self._initialized:
            return
        self.makeCurrent()
        for query in self._queries:
            query.destroy()
        for buffer in (self.vbo, self.triangle_buffer, self.edge_buffer):
            buffer.destroy()
        self.vao.destroy()
        self.program.removeAllShaders()
        self._initialized = False
        self._queries.clear()
        self._available_queries.clear()
        self._pending_queries.clear()
        self.doneCurrent()
