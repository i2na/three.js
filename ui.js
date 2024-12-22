import { vec2, vec3, vec4, mat4, utils } from "wgpu-matrix";
import { unproject_vector } from "./utils.js";

export class UI {
  static NONE = 0;
  static ROTATING = 1;
  static TRANSLATING = 2;
  static mouseMove = UI.NONE;
  static camera = {
    fovy: 45,
    position: vec3.create(-0.2, -0.5, -4),
    near: 0.1,
    far: 100,
  };
  static matrices = { P: null, R: null, VP: null };

  static onmousedown(ev) {
    if (ev.buttons === 1) {
      if (ev.metaKey || ev.ctrlKey) UI.mouseMove = UI.TRANSLATING;
      else UI.mouseMove = UI.ROTATING;
    }
  }
  static onmouseup(ev) {
    UI.mouseMove = UI.NONE;
  }
  static onmousemove(ev) {
    let offset = [ev.movementX, ev.movementY];
    if (UI.mouseMove == UI.ROTATING) {
      UI.update_VP();
      let axis = unproject_vector([offset[1], offset[0], 0], UI.matrices.VP, [
        0,
        0,
        UI.canvas.clientWidth,
        UI.canvas.clientHeight,
      ]);
      UI.matrices.R = mat4.rotate(
        UI.matrices.R,
        [axis[0], axis[1], axis[2]],
        utils.degToRad(vec2.lenSq(offset) * 0.1)
      );
    } else if (UI.mouseMove == UI.TRANSLATING) {
      UI.update_VP();
      let by = unproject_vector([offset[0], -offset[1], 0], UI.matrices.VP, [
        0,
        0,
        UI.canvas.clientWidth,
        UI.canvas.clientHeight,
      ]);
      UI.camera.position = vec3.add(
        UI.camera.position,
        vec3.transformMat4(vec3.create(by[0], by[1], by[2]), UI.matrices.R)
      );
    }
  }
  static onwheel(ev) {
    ev.preventDefault();
    UI.camera.position[2] = -Math.max(
      1,
      Math.min(-UI.camera.position[2] + ev.deltaY * 0.01, 50)
    );
    UI.update_VP();
  }
  static update_VP() {
    UI.matrices.P = mat4.perspective(
      utils.degToRad(UI.camera.fovy),
      UI.canvas.width / UI.canvas.height,
      UI.camera.near,
      UI.camera.far
    );
    if (!UI.matrices.R) UI.matrices.R = mat4.identity();
    let T = mat4.translate(UI.matrices.P, UI.camera.position);
    UI.matrices.VP = mat4.multiply(T, UI.matrices.R);
  }
}
