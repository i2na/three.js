import { computeTangents } from "./utils.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";

export async function loadAllMeshesFromGLTF(device, url) {
  const loader = new GLTFLoader();
  const gltf = await new Promise((resolve, reject) => {
    loader.load(
      url,
      (g) => resolve(g),
      undefined,
      (e) => reject(e)
    );
  });

  let meshes = [];
  gltf.scene.traverse((c) => {
    if (c.isMesh && c.geometry) meshes.push(c);
  });
  if (meshes.length == 0) throw Error("No meshes found in gltf");

  function createBufferFromArray(arr, usage) {
    const buf = device.createBuffer({
      size: arr.byteLength,
      usage: usage | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, arr);
    return buf;
  }

  let meshData = [];
  for (let obj of meshes) {
    const positions = obj.geometry.attributes.position.array;

    const normals = obj.geometry.attributes.normal.array;
    const uvs = obj.geometry.attributes.uv
      ? obj.geometry.attributes.uv.array
      : new Float32Array((positions.length / 3) * 2);
    let indices = obj.geometry.index.array;
    if (!(indices instanceof Uint32Array)) {
      const tmp = new Uint32Array(indices.length);
      for (let i = 0; i < indices.length; i++) tmp[i] = indices[i];
      indices = tmp;
    }

    const { tangents, bitangents } = computeTangents(
      positions,
      normals,
      uvs,
      indices
    );

    const posBuffer = createBufferFromArray(positions, GPUBufferUsage.VERTEX);
    const normalBuffer = createBufferFromArray(normals, GPUBufferUsage.VERTEX);
    const uvBuffer = createBufferFromArray(uvs, GPUBufferUsage.VERTEX);
    const tangentBuffer = createBufferFromArray(
      tangents,
      GPUBufferUsage.VERTEX
    );
    const bitangentBuffer = createBufferFromArray(
      bitangents,
      GPUBufferUsage.VERTEX
    );
    const indexBuffer = createBufferFromArray(indices, GPUBufferUsage.INDEX);
    meshData.push({
      posBuffer,
      normalBuffer,
      uvBuffer,
      tangentBuffer,
      bitangentBuffer,
      indexBuffer,
      indexCount: indices.length,
    });
  }
  return meshData;
}

export async function loadOBJAsMeshes(device, url) {
  const loader = new OBJLoader();
  const obj = await new Promise((resolve, reject) => {
    loader.load(
      url,
      (obj) => resolve(obj),
      undefined,
      (e) => reject(e)
    );
  });
  let meshes = [];
  obj.traverse((c) => {
    if (c.isMesh && c.geometry) meshes.push(c);
  });
  if (meshes.length == 0) throw Error("No meshes found in obj");

  function createBufferFromArray(arr, usage) {
    const buf = device.createBuffer({
      size: arr.byteLength,
      usage: usage | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, arr);
    return buf;
  }

  let meshData = [];
  for (let m of meshes) {
    const geom = m.geometry;
    const positions = geom.attributes.position.array;
    const normals = geom.attributes.normal.array;
    let uvs;
    if (geom.attributes.uv) uvs = geom.attributes.uv.array;
    else uvs = new Float32Array((positions.length / 3) * 2);
    let indices;
    if (geom.index) indices = geom.index.array;
    else {
      let count = positions.length / 3;
      indices = new Uint32Array(count);
      for (let i = 0; i < count; i++) indices[i] = i;
    }

    const { tangents, bitangents } = computeTangents(
      positions,
      normals,
      uvs,
      indices
    );

    const posBuffer = createBufferFromArray(positions, GPUBufferUsage.VERTEX);
    const normalBuffer = createBufferFromArray(normals, GPUBufferUsage.VERTEX);
    const uvBuffer = createBufferFromArray(uvs, GPUBufferUsage.VERTEX);
    const tangentBuffer = createBufferFromArray(
      tangents,
      GPUBufferUsage.VERTEX
    );
    const bitangentBuffer = createBufferFromArray(
      bitangents,
      GPUBufferUsage.VERTEX
    );
    const indexBuffer = createBufferFromArray(indices, GPUBufferUsage.INDEX);
    meshData.push({
      posBuffer,
      normalBuffer,
      uvBuffer,
      tangentBuffer,
      bitangentBuffer,
      indexBuffer,
      indexCount: indices.length,
    });
  }
  return meshData;
}

export async function loadImageAsTexture(device, url) {
  const img = new Image();
  img.src = url;
  await img.decode();
  const bitmap = await createImageBitmap(img, {
    colorSpaceConversion: "none",
  });
  const texture = device.createTexture({
    label: url,
    format: "rgba8unorm",
    size: [bitmap.width, bitmap.height],
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.copyExternalImageToTexture(
    { source: bitmap },
    { texture: texture },
    [bitmap.width, bitmap.height]
  );
  return texture;
}
