import mujoco
import glfw
import numpy as np
import os
from OpenGL.GL import *
import time

class MujocoViewer:
    def __init__(self, xml_path):
        # 加载 MuJoCo 模型
        xml_path = os.path.abspath(xml_path)
        print(f"加载 XML: {xml_path}")
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML 文件未找到: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)

        # 初始化 GLFW
        if not glfw.init():
            raise RuntimeError("GLFW 初始化失败")
        self.window = glfw.create_window(640, 480, "MuJoCo Viewer", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW 窗口创建失败")
        glfw.make_context_current(self.window)
        glfw.show_window(self.window)
        glfw.focus_window(self.window)
        print("GLFW 窗口创建成功")

        # 初始化相机和交互
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance = 3.0
        self.cam.lookat = np.array([0.0, 0.0, 0.7])
        self.cam.azimuth = 150
        self.cam.elevation = -20

        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        # 鼠标交互变量
        self.lastx = 0
        self.lasty = 0
        self.button_left = False
        self.button_middle = False
        self.button_right = False

        # 设置 GLFW 回调
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_scroll_callback(self.window, self.mouse_scroll)

    def mouse_button(self, window, button, action, mods):
        self.button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self.button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        self.button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
        x, y = glfw.get_cursor_pos(window)
        self.lastx, self.lasty = x, y

    def mouse_move(self, window, x, y):
        dx = x - self.lastx
        dy = y - self.lasty
        self.lastx, self.lasty = x, y
        if not (self.button_left or self.button_middle or self.button_right):
            return
        width, height = glfw.get_window_size(window)
        mod_shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                     glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        if self.button_right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif self.button_left:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(self.model, action, dx/height, dy/height, self.scene, self.cam)

    def mouse_scroll(self, window, x_offset, y_offset):
        action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(self.model, action, 0.0, -0.05 * y_offset, self.scene, self.cam)

    def render(self):
        if not self.window or glfw.window_should_close(self.window):
            return False
        self.renderer.update_scene(self.data, self.cam)
        viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(self.window))
        mujoco.mjr_render(viewport, self.scene, self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        return True

    def run(self):
        while self.render():
            mujoco.mj_step(self.model, self.data)
            time.sleep(0.01)  # 控制帧率
        self.close()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
        if self.context is not None:
            self.context.free()
        if self.window is not None:
            glfw.destroy_window(self.window)
            glfw.terminate()

if __name__ == "__main__":
    xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "zhiyuan_scene.xml"))
    viewer = MujocoViewer(xml_path)
    viewer.run()