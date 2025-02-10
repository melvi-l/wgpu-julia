import './style.css'

const WORKGROUP_SIZE = 8

async function main(...size) {

    if (!navigator.gpu) {
        alert("no webGPU")
    }

    const adapter = await navigator.gpu.requestAdapter();

    if (!adapter) {
        alert("no GPU adapter")
    }

    const device = await adapter.requestDevice()
    console.log("using device:", device)
    device.addEventListener("uncapturederror", ev => console.log(ev.error.message))

    const format = navigator.gpu.getPreferredCanvasFormat()
    console.log("using format:", format)

    const canvas = document.getElementById("webgpu")
    canvas.width = size[0]
    canvas.height = size[1]
    const ctx = canvas.getContext("webgpu")
    ctx.configure({
        device,
        format
    })
    const uniformSize = 32
    const uniformBuffer = device.createBuffer({
        label: "Julia set parameters buffer",
        size: uniformSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    })
    const uniformArray = new ArrayBuffer(uniformSize);
    const uniformFloats = new Float32Array(uniformArray);
    const uniformUint32 = new Uint32Array(uniformArray);
    function setUniform({
        min,
        max,
        c,
        maxIteration
    }) {
        uniformFloats[0] = min.x;
        uniformFloats[1] = min.y;
        uniformFloats[2] = max.x;
        uniformFloats[3] = max.y;
        uniformFloats[4] = c.x;
        uniformFloats[5] = c.y;
        uniformUint32[6] = maxIteration;

        device.queue.writeBuffer(uniformBuffer, 0, uniformArray)
    }
    setUniform({
        min: {
            x: -1.1,
            y: -1.1
        },
        max: {
            x: 1.1,
            y: 1.1
        },
        c: { // {-.7, .27015}
            x: -0.8,
            y: 0.156
        },
        maxIteration: 100
    })

    /**
     * COMPUTE 
     */
    const computedTexture = device.createTexture({
        size,
        format: "rgba8unorm",
        usage: GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_SRC |
            GPUTextureUsage.TEXTURE_BINDING,
    });
    const computeShaderModule = device.createShaderModule({
        label: "Julia set compute shader",
        code: `
        struct JuliaParams {
            minX: f32,
            minY: f32,
            maxX: f32,
            maxY: f32,
            c: vec2<f32>,
            maxIterations: u32,
        }

        @group(0) @binding(0) 
        var<uniform> params: JuliaParams;

        @group(0) @binding(1)
        var result: texture_storage_2d<rgba8unorm, write>;

        @compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
        fn computeMain (@builtin(global_invocation_id) gid: vec3<u32>) {
            let dims = textureDimensions(result);
            if (gid.x >= dims.x || gid.y >= dims.y) {
              return;
            }

            let x0 = params.minX + (f32(gid.x) / f32(dims.x)) * (params.maxX - params.minX);
            let y0 = params.minY + (f32(gid.y) / f32(dims.y)) * (params.maxY - params.minY);

            var x = x0;
            var y = y0;
            var iter: u32 = 0u;

            while ((x * x + y * y <= 4.0) && (iter < params.maxIterations)) {
              let xtemp = x * x - y * y + params.c.x;
              y = 2.0 * x * y + params.c.y;
              x = xtemp;
              iter = iter + 1u;
            }

            let normalized = f32(iter) / f32(params.maxIterations);
            let color = vec4<f32>(normalized, normalized, normalized, 1.0);
            textureStore(result, vec2<i32>(gid.xy), color);
        }
        `
    })

    const computePipeline = device.createComputePipeline({
        label: "Julia set compute pipeline",
        layout: "auto",
        compute: {
            module: computeShaderModule,
            entryPoint: "computeMain"
        }
    })


    const computeBindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: computedTexture.createView() },
        ],
    });


    /**
     * RENDER 
     */
    const vertexModule = device.createShaderModule({
        label: "Julia set vertex shader",
        code: `
        @vertex
        fn vertexMain(@builtin(vertex_index) VertexIndex : u32) -> @builtin(position) vec4<f32> {
          var pos = array<vec2<f32>, 6>(
            vec2<f32>(-1.0, -1.0),
            vec2<f32>(1.0, -1.0),
            vec2<f32>(-1.0, 1.0),
            vec2<f32>(-1.0, 1.0),
            vec2<f32>(1.0, -1.0),
            vec2<f32>(1.0, 1.0)
          );
          return vec4<f32>(pos[VertexIndex], 0.0, 1.0);
        }
        `
    })

    const fragmentModule = device.createShaderModule({
        label: "Julia set fragment shader",
        code: `
        @group(0) @binding(0)
        var juliaTexture: texture_2d<f32>;
        @group(0) @binding(1)
        var sampler0: sampler;

        @fragment
        fn fragmentMain(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
          let dims = textureDimensions(juliaTexture);
          let uv = fragCoord.xy / vec2<f32>(dims);
          return textureSample(juliaTexture, sampler0, uv);
        }    
        `
    })

    const renderPipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: {
            module: vertexModule,
            entryPoint: "vertexMain",
        },
        fragment: {
            module: fragmentModule,
            entryPoint: "fragmentMain",
            targets: [{ format }],
        },
        primitive: {
            topology: "triangle-list",
        },
    });

    const sampler = device.createSampler({
        magFilter: "linear",
        minFilter: "linear",
    });

    const renderBindGroup = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: computedTexture.createView() },
            { binding: 1, resource: sampler },
        ],
    });


    function frame() {
        const commandEncoder = device.createCommandEncoder();

        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(computePipeline);
        computePass.setBindGroup(0, computeBindGroup);
        const workgroupCountX = Math.ceil(size[0] / 8);
        const workgroupCountY = Math.ceil(size[1] / 8);
        computePass.dispatchWorkgroups(workgroupCountX, workgroupCountY);
        computePass.end();

        const currentTexture = ctx.getCurrentTexture();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: currentTexture.createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: "clear",
                storeOp: "store",
            }],
        });
        renderPass.setPipeline(renderPipeline);
        renderPass.setBindGroup(0, renderBindGroup);
        renderPass.draw(6, 1, 0, 0);
        renderPass.end();

        device.queue.submit([commandEncoder.finish()]);
    }

    frame()

}

main(window.innerWidth, window.innerHeight)


