export function createSampler(device) {
  return device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    mipmapFilter: "linear",
    maxAnisotropy: 16,
  });
}

export function createDepthTexture(device, width, height, format) {
  return device.createTexture({
    size: [width, height],
    format: format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
}

export function createRenderPipeline(
  device,
  vertexModule,
  fragmentModule,
  depthFormat,
  preferredFormat
) {
  return device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: vertexModule,
      entryPoint: "main",
      buffers: [
        {
          arrayStride: 4 * 3,
          attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
        },
        {
          arrayStride: 4 * 3,
          attributes: [{ shaderLocation: 1, offset: 0, format: "float32x3" }],
        },
        {
          arrayStride: 4 * 2,
          attributes: [{ shaderLocation: 2, offset: 0, format: "float32x2" }],
        },
        {
          arrayStride: 4 * 3,
          attributes: [{ shaderLocation: 3, offset: 0, format: "float32x3" }],
        },
        {
          arrayStride: 4 * 3,
          attributes: [{ shaderLocation: 4, offset: 0, format: "float32x3" }],
        },
      ],
    },
    fragment: {
      module: fragmentModule,
      entryPoint: "main",
      targets: [{ format: preferredFormat }],
    },
    primitive: { topology: "triangle-list" },
    depthStencil: {
      format: depthFormat,
      depthWriteEnabled: true,
      depthCompare: "less",
    },
  });
}
