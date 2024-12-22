import * as THREE from "three";

import { vec2, vec3, vec4, mat4, utils } from "wgpu-matrix";
import { vertexShaderWGSL } from "./shaders/vertex.js";
import { fragmentShaderWGSL } from "./shaders/fragment.js";
import { gridVertexCode } from "./shaders/gridVertex.js";
import { localAxesVertexCode } from "./shaders/localAxesVertex.js";
import { localAxesFragmentCode } from "./shaders/localAxesFragment.js";
import { UI } from "./ui.js";
import {
  createSampler,
  createDepthTexture,
  createRenderPipeline,
} from "./pipelines.js";
import {
  project,
  unproject,
  unproject_vector,
  computeTangents,
} from "./utils.js";
import {
  loadAllMeshesFromGLTF,
  loadOBJAsMeshes,
  loadImageAsTexture,
} from "./loader.js";

async function main() {
  const canvas = document.getElementById("webgpu-canvas");
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu");
  const preferredFormat = navigator.gpu.getPreferredCanvasFormat();

  const vertexModule = device.createShaderModule({
    code: vertexShaderWGSL,
  });
  const fragmentModule = device.createShaderModule({
    code: fragmentShaderWGSL,
  });
  const gridModule = device.createShaderModule({ code: gridVertexCode });
  const localAxesVertexModule = device.createShaderModule({
    code: localAxesVertexCode,
  });
  const localAxesFragmentModule = device.createShaderModule({
    code: localAxesFragmentCode,
  });

  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  context.configure({
    device: device,
    format: preferredFormat,
    alphaMode: "opaque",
  });

  canvas.onmousedown = UI.onmousedown;
  canvas.onmouseup = UI.onmouseup;
  canvas.onmousemove = UI.onmousemove;
  window.addEventListener("wheel", UI.onwheel, { passive: false });

  UI.canvas = canvas;
  UI.update_VP();

  const meshData = await loadAllMeshesFromGLTF(device, "models/tank.glb");
  const shellData = await loadOBJAsMeshes(device, "models/shell.obj");

  const tankColor = await loadImageAsTexture(
    device,
    "textures/tank-color.jpeg"
  );
  const tankEtc = await loadImageAsTexture(device, "textures/tank-etc.png");
  const tankNormal = await loadImageAsTexture(
    device,
    "textures/tank-normal.png"
  );

  const shellColor = await loadImageAsTexture(
    device,
    "textures/shell-color.png"
  );
  const shellEtc = await loadImageAsTexture(device, "textures/shell-etc.png");
  const shellNormal = await loadImageAsTexture(
    device,
    "textures/shell-normal.png"
  );

  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    mipmapFilter: "linear",
    maxAnisotropy: 16,
  });

  const sceneBuffer = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const modelBuffer = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const modelBuffer_turret = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const modelBuffer_barrel = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const normalMatrixBuffer = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const normalMatrixBuffer_turret = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const normalMatrixBuffer_barrel = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const lightBuffer = device.createBuffer({
    size: 12,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  function createPartIDBuffer(id) {
    const buf = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint32Array(buf.getMappedRange())[0] = id;
    buf.unmap();
    return buf;
  }

  const shells = [];

  function spawnShell(initialPos, initialVel) {
    let shell = {
      pos: [...initialPos],
      vel: [...initialVel],
      alive: true,

      modelBuffer: device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      normalMatrixBuffer: device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
    };

    shell.bindGroup = device.createBindGroup({
      layout: pipelineShell.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: sceneBuffer } },
        { binding: 1, resource: { buffer: shell.modelBuffer } },
        { binding: 2, resource: { buffer: shell.normalMatrixBuffer } },
        { binding: 3, resource: { buffer: lightBuffer } },
        { binding: 4, resource: shellColor.createView() },
        { binding: 5, resource: shellEtc.createView() },
        { binding: 6, resource: shellNormal.createView() },
        { binding: 7, resource: sampler },
        { binding: 8, resource: { buffer: partIDBuffer_Shell } },
      ],
    });

    shells.push(shell);
  }

  // partIDs: 0=body,1=turret,2=innerWheel,3=outerWheel,4=barrel,5=shell
  const partIDBuffer_Body = createPartIDBuffer(0);
  const partIDBuffer_Turret = createPartIDBuffer(1);
  const partIDBuffer_InnerWheel = createPartIDBuffer(2);
  const partIDBuffer_OuterWheel = createPartIDBuffer(3);
  const partIDBuffer_Barrel = createPartIDBuffer(4);
  const partIDBuffer_Shell = createPartIDBuffer(5);

  const pipeline = createRenderPipeline(
    device,
    vertexModule,
    fragmentModule,
    "depth24plus",
    preferredFormat
  );

  const pipelineShell = createRenderPipeline(
    device,
    vertexModule,
    fragmentModule,
    "depth24plus",
    preferredFormat
  );

  const pipelineTurret = createRenderPipeline(
    device,
    vertexModule,
    fragmentModule,
    "depth24plus",
    preferredFormat
  );

  const pipelineBarrel = createRenderPipeline(
    device,
    vertexModule,
    fragmentModule,
    "depth24plus",
    preferredFormat
  );

  function createBindGroup(colorTex, etcTex, normalTex, partIDBuf) {
    return device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: sceneBuffer } },
        { binding: 1, resource: { buffer: modelBuffer } },
        { binding: 2, resource: { buffer: normalMatrixBuffer } },
        { binding: 3, resource: { buffer: lightBuffer } },
        { binding: 4, resource: colorTex.createView() },
        { binding: 5, resource: etcTex.createView() },
        { binding: 6, resource: normalTex.createView() },
        { binding: 7, resource: sampler },
        { binding: 8, resource: { buffer: partIDBuf } },
      ],
    });
  }

  const bindGroup_Body = createBindGroup(
    tankColor,
    tankEtc,
    tankNormal,
    partIDBuffer_Body
  );
  const bindGroup_Turret = device.createBindGroup({
    layout: pipelineTurret.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: sceneBuffer } },
      { binding: 1, resource: { buffer: modelBuffer_turret } },
      { binding: 2, resource: { buffer: normalMatrixBuffer_turret } },
      { binding: 3, resource: { buffer: lightBuffer } },
      { binding: 4, resource: tankColor.createView() },
      { binding: 5, resource: tankEtc.createView() },
      { binding: 6, resource: tankNormal.createView() },
      { binding: 7, resource: sampler },
      { binding: 8, resource: { buffer: partIDBuffer_Turret } },
    ],
  });
  const bindGroup_InnerWheel = createBindGroup(
    tankColor,
    tankEtc,
    tankNormal,
    partIDBuffer_InnerWheel
  );
  const bindGroup_OuterWheel = createBindGroup(
    tankColor,
    tankEtc,
    tankNormal,
    partIDBuffer_OuterWheel
  );
  const bindGroup_Barrel = device.createBindGroup({
    layout: pipelineBarrel.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: sceneBuffer } },
      { binding: 1, resource: { buffer: modelBuffer_barrel } },
      { binding: 2, resource: { buffer: normalMatrixBuffer_barrel } },
      { binding: 3, resource: { buffer: lightBuffer } },
      { binding: 4, resource: tankColor.createView() },
      { binding: 5, resource: tankEtc.createView() },
      { binding: 6, resource: tankNormal.createView() },
      { binding: 7, resource: sampler },
      { binding: 8, resource: { buffer: partIDBuffer_Barrel } },
    ],
  });

  const gridPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: gridModule,
      entryPoint: "main",
      buffers: [
        {
          arrayStride: 24,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x3" },
          ],
        },
      ],
    },
    fragment: {
      module: gridModule,
      entryPoint: "fs_main",
      targets: [{ format: preferredFormat }],
    },
    primitive: { topology: "line-list" },
    depthStencil: {
      format: "depth24plus",
      depthWriteEnabled: true,
      depthCompare: "less",
    },
  });

  const { buffer: gridBuffer, vertexCount: gridVertexCount } =
    (function createGridAndAxesBuffers(device) {
      const lines = [];
      lines.push(...[0, 0, 0, 1, 0, 0]);
      lines.push(...[1, 0, 0, 1, 0, 0]);
      lines.push(...[0, 0, 0, 0, 1, 0]);
      lines.push(...[0, 1, 0, 0, 1, 0]);
      lines.push(...[0, 0, 0, 0, 0, 1]);
      lines.push(...[0, 0, 1, 0, 0, 1]);

      for (let i = -5; i <= 5; i++) {
        lines.push(...[-5, 0, i, 1.0, 1.0, 1.0]);
        lines.push(...[5, 0, i, 1.0, 1.0, 1.0]);
        lines.push(...[i, 0, -5, 1.0, 1.0, 1.0]);
        lines.push(...[i, 0, 5, 1.0, 1.0, 1.0]);
      }

      const data = new Float32Array(lines);
      const buf = device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buf, 0, data);
      return { buffer: buf, vertexCount: data.length / 6 };
    })(device);

  const gridBindGroup = device.createBindGroup({
    layout: gridPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: sceneBuffer } }],
  });

  const localAxesPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: localAxesVertexModule,
      entryPoint: "main",
      buffers: [
        {
          arrayStride: 24,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x3" },
          ],
        },
      ],
    },
    fragment: {
      module: localAxesFragmentModule,
      entryPoint: "fs_main",
      targets: [{ format: preferredFormat }],
    },
    primitive: { topology: "line-list" },
    depthStencil: {
      format: "depth24plus",
      depthWriteEnabled: true,
      depthCompare: "less",
    },
  });

  const { buffer: localAxesBuffer, vertexCount: localAxesVertexCount } =
    (function createLocalAxesBuffer(device) {
      const axesLines = [];
      axesLines.push(...[0, 0, 0, 1, 0, 0]);
      axesLines.push(...[0.5, 0, 0, 1, 0, 0]);
      axesLines.push(...[0, 0, 0, 0, 1, 0]);
      axesLines.push(...[0, 0.5, 0, 0, 1, 0]);
      axesLines.push(...[0, 0, 0, 0, 0, 1]);
      axesLines.push(...[0, 0, 0.5, 0, 0, 1]);

      const data = new Float32Array(axesLines);
      const buf = device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buf, 0, data);
      return { buffer: buf, vertexCount: axesLines.length / 6 };
    })(device);

  const localAxesBindGroup = device.createBindGroup({
    layout: localAxesPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: sceneBuffer } },
      { binding: 1, resource: { buffer: modelBuffer } },
    ],
  });

  const tankState = {
    position: [0, 0, 0],
    rotation: 0,
    turretRotation: 0,
    barrelElevation: 0,
  };

  let turretPivot = [0.45, 0.299, 0];
  let barrelPivot = [0.65, 0.87, 0];
  let joinPivot = [0.2, 0, 0];

  function computeBarrelPivot(tankState) {
    let M_body = mat4.identity();
    M_body = mat4.translate(M_body, [
      tankState.position[0],
      0.46,
      tankState.position[2],
    ]);
    M_body = mat4.rotateY(M_body, tankState.rotation);

    let M_turret = mat4.clone(M_body);
    M_turret = mat4.translate(M_turret, [
      -turretPivot[0],
      -turretPivot[1],
      -turretPivot[2],
    ]);
    M_turret = mat4.rotateY(M_turret, tankState.turretRotation);
    M_turret = mat4.translate(M_turret, turretPivot);

    let M_barrel = mat4.clone(M_turret);
    M_barrel = mat4.translate(M_barrel, [
      -joinPivot[0],
      -joinPivot[1],
      -joinPivot[2],
    ]);
    M_barrel = mat4.rotate(M_barrel, [0, 0, 1], tankState.barrelElevation);
    M_barrel = mat4.translate(M_barrel, joinPivot);
    let localEnd = [barrelPivot[0], barrelPivot[1] - 0.46, barrelPivot[2], 1];
    let worldEnd = vec4.transformMat4(localEnd, M_barrel);

    return [worldEnd[0], worldEnd[1], worldEnd[2]];
  }

  let shellPaused = false;

  window.addEventListener("keydown", (e) => {
    let forwardX = Math.cos(tankState.rotation);
    let forwardZ = Math.sin(tankState.rotation);

    const key = e.key.toLowerCase();
    switch (key) {
      case "arrowup":
        tankState.position[0] += forwardX * 0.1;
        tankState.position[2] -= forwardZ * 0.1;
        break;
      case "arrowdown":
        tankState.position[0] -= forwardX * 0.1;
        tankState.position[2] += forwardZ * 0.1;
        break;
      case "arrowleft":
        tankState.rotation += 0.05;
        break;
      case "arrowright":
        tankState.rotation -= 0.05;
        break;
      case "a":
        tankState.turretRotation += 0.05;
        break;
      case "d":
        tankState.turretRotation -= 0.05;
        break;
      case "w":
        if (tankState.barrelElevation < 0.1) tankState.barrelElevation += 0.05;
        break;
      case "s":
        if (tankState.barrelElevation > -0.1) tankState.barrelElevation -= 0.05;
        break;
      case " ":
        const barrel_pivot = computeBarrelPivot(tankState);

        let forward_X = Math.cos(tankState.rotation + tankState.turretRotation);
        let forward_Z = Math.sin(tankState.rotation + tankState.turretRotation);

        let speed = 1.3;
        let dx = forward_X * speed;
        let dy = 0.12;
        let dz = -forward_Z * speed;

        let initialVel = [dx, dy, dz];

        spawnShell(
          [barrel_pivot[0], barrel_pivot[1], barrel_pivot[2]],
          initialVel
        );
        break;
      case "p":
        shellPaused = !shellPaused;
        console.log("shellPaused =", shellPaused);
        break;
    }
  });

  function updateAllShells() {
    let dt = 1 / 60;
    for (let shell of shells) {
      if (!shell.alive) continue;

      if (shellPaused) {
        continue;
      }

      shell.vel[1] -= 9.8 * 0.06 * dt;

      shell.pos[0] += shell.vel[0] * dt;
      shell.pos[1] += shell.vel[1] * dt;
      shell.pos[2] += shell.vel[2] * dt;

      if (shell.pos[1] < 0) {
        shell.pos[1] = 0;
        shell.alive = false;
      }
    }
  }

  function drawPart(renderPass, M, bindGroup, mesh) {
    let invM = mat4.invert(M);
    let normalMat = mat4.transpose(invM);
    device.queue.writeBuffer(modelBuffer, 0, M);
    device.queue.writeBuffer(normalMatrixBuffer, 0, normalMat);

    renderPass.setPipeline(pipeline);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.setVertexBuffer(0, mesh.posBuffer);
    renderPass.setVertexBuffer(1, mesh.normalBuffer);
    renderPass.setVertexBuffer(2, mesh.uvBuffer);
    renderPass.setVertexBuffer(3, mesh.tangentBuffer);
    renderPass.setVertexBuffer(4, mesh.bitangentBuffer);
    renderPass.setIndexBuffer(mesh.indexBuffer, "uint32");
    renderPass.drawIndexed(mesh.indexCount, 1, 0, 0, 0);
  }

  function renderFrame(time) {
    updateAllShells();

    let M_body = mat4.identity();
    M_body = mat4.translate(M_body, [
      tankState.position[0],
      0.46,
      tankState.position[2],
    ]);
    M_body = mat4.rotate(M_body, [0, 1, 0], tankState.rotation);

    let M_turret = mat4.clone(M_body);
    M_turret = mat4.translate(M_turret, [
      -turretPivot[0],
      -turretPivot[1],
      -turretPivot[2],
    ]);
    M_turret = mat4.rotate(M_turret, [0, 1, 0], tankState.turretRotation);
    M_turret = mat4.translate(M_turret, [
      turretPivot[0],
      turretPivot[1],
      turretPivot[2],
    ]);

    let M_barrel = mat4.clone(M_turret);
    M_barrel = mat4.translate(M_barrel, [
      -joinPivot[0],
      -joinPivot[1],
      -joinPivot[2],
    ]);

    M_barrel = mat4.rotate(M_barrel, [0, 0, 1], tankState.barrelElevation);
    M_barrel = mat4.translate(M_barrel, joinPivot);

    let M_innerWheel = M_body;
    let M_outerWheel = M_body;

    const lightDir = new Float32Array([-1, 5, 5]);
    device.queue.writeBuffer(lightBuffer, 0, lightDir);

    device.queue.writeBuffer(sceneBuffer, 0, UI.matrices.VP);

    const depthTexture = createDepthTexture(
      device,
      canvas.width,
      canvas.height,
      "depth24plus"
    );
    const commandEncoder = device.createCommandEncoder();
    const textureView = context.getCurrentTexture().createView();

    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: textureView,
          loadOp: "clear",
          clearValue: { r: 0.3, g: 0.3, b: 0.3, a: 1 },
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthLoadOp: "clear",
        depthClearValue: 1.0,
        depthStoreOp: "store",
      },
    });

    renderPass.setPipeline(gridPipeline);
    renderPass.setBindGroup(0, gridBindGroup);
    renderPass.setVertexBuffer(0, gridBuffer);
    renderPass.draw(gridVertexCount, 1, 0, 0);

    drawPart(renderPass, M_body, bindGroup_Body, meshData[0]);

    let invM_Turret = mat4.invert(M_turret);
    let normalMat_Turret = mat4.transpose(invM_Turret);
    device.queue.writeBuffer(modelBuffer_turret, 0, M_turret);
    device.queue.writeBuffer(normalMatrixBuffer_turret, 0, normalMat_Turret);
    renderPass.setPipeline(pipelineTurret);
    renderPass.setBindGroup(0, bindGroup_Turret);
    renderPass.setVertexBuffer(0, meshData[1].posBuffer);
    renderPass.setVertexBuffer(1, meshData[1].normalBuffer);
    renderPass.setVertexBuffer(2, meshData[1].uvBuffer);
    renderPass.setVertexBuffer(3, meshData[1].tangentBuffer);
    renderPass.setVertexBuffer(4, meshData[1].bitangentBuffer);
    renderPass.setIndexBuffer(meshData[1].indexBuffer, "uint32");
    renderPass.drawIndexed(meshData[1].indexCount, 1, 0, 0, 0);

    drawPart(renderPass, M_innerWheel, bindGroup_InnerWheel, meshData[2]);
    drawPart(renderPass, M_outerWheel, bindGroup_OuterWheel, meshData[3]);

    let invM_Barrel = mat4.invert(M_barrel);
    let normalMat_Barrel = mat4.transpose(invM_Barrel);
    device.queue.writeBuffer(modelBuffer_barrel, 0, M_barrel);
    device.queue.writeBuffer(normalMatrixBuffer_barrel, 0, normalMat_Barrel);
    renderPass.setPipeline(pipelineBarrel);
    renderPass.setBindGroup(0, bindGroup_Barrel);
    renderPass.setVertexBuffer(0, meshData[4].posBuffer);
    renderPass.setVertexBuffer(1, meshData[4].normalBuffer);
    renderPass.setVertexBuffer(2, meshData[4].uvBuffer);
    renderPass.setVertexBuffer(3, meshData[4].tangentBuffer);
    renderPass.setVertexBuffer(4, meshData[4].bitangentBuffer);
    renderPass.setIndexBuffer(meshData[4].indexBuffer, "uint32");
    renderPass.drawIndexed(meshData[4].indexCount, 1, 0, 0, 0);

    for (let shell of shells) {
      if (!shell.alive) continue;

      let M_shell_t = mat4.identity();
      M_shell_t = mat4.translate(M_shell_t, shell.pos);

      let [vx, vy, vz] = shell.vel;
      let speed = Math.sqrt(vx * vx + vy * vy + vz * vz);
      if (speed > 1e-6) {
        vx /= speed;
        vy /= speed;
        vz /= speed;
      }

      let yaw = Math.atan2(-vz, vx);
      let pitch = Math.atan2(vy, Math.sqrt(vx * vx + vz * vz));

      M_shell_t = mat4.rotate(M_shell_t, [0, 1, 0], yaw);
      M_shell_t = mat4.rotate(M_shell_t, [0, 0, 1], pitch);

      M_shell_t = mat4.scale(M_shell_t, [1.1, 1.1, 1.1]);

      let M_shell_inv = mat4.invert(M_shell_t);
      let M_shell_nrm = mat4.transpose(M_shell_inv);

      device.queue.writeBuffer(shell.modelBuffer, 0, M_shell_t);
      device.queue.writeBuffer(shell.normalMatrixBuffer, 0, M_shell_nrm);

      renderPass.setPipeline(pipelineShell);
      renderPass.setBindGroup(0, shell.bindGroup);

      renderPass.setVertexBuffer(0, shellData[0].posBuffer);
      renderPass.setVertexBuffer(1, shellData[0].normalBuffer);
      renderPass.setVertexBuffer(2, shellData[0].uvBuffer);
      renderPass.setVertexBuffer(3, shellData[0].tangentBuffer);
      renderPass.setVertexBuffer(4, shellData[0].bitangentBuffer);
      renderPass.setIndexBuffer(shellData[0].indexBuffer, "uint32");
      renderPass.drawIndexed(shellData[0].indexCount, 1, 0, 0, 0);
    }

    device.queue.writeBuffer(modelBuffer, 0, M_body);
    renderPass.setPipeline(localAxesPipeline);
    renderPass.setBindGroup(0, localAxesBindGroup);
    renderPass.setVertexBuffer(0, localAxesBuffer);
    renderPass.draw(localAxesVertexCount, 1, 0, 0);

    renderPass.end();
    device.queue.submit([commandEncoder.finish()]);
    requestAnimationFrame(renderFrame);
  }

  requestAnimationFrame(renderFrame);

  window.addEventListener("resize", () => {
    canvas.width = canvas.clientWidth * devicePixelRatio;
    canvas.height = canvas.clientHeight * devicePixelRatio;
    context.configure({ device, format: preferredFormat });
  });
}

main();
