import * as tf from '@tensorflow/tfjs'
import { Root } from 'protobufjs'

const onnxProto = `
syntax = "proto3";
package onnx;
message TensorProto {
  enum DataType { UNDEFINED=0; FLOAT=1; INT8=3; INT32=6; DOUBLE=11; }
  DataType data_type = 2;
  repeated int64 dims = 7;
  bytes raw_data = 13;
}
message AttributeProto { string name=1; string type=2; repeated int64 ints=6; repeated float floats=7; }
message NodeProto { string op_type=3; repeated string input=5; repeated string output=6; repeated AttributeProto attribute=7; }
message ValueInfoProto { string name=1; }
message GraphProto { string name=1; repeated NodeProto node=2; repeated ValueInfoProto input=4; repeated ValueInfoProto output=5; repeated TensorProto initializer=7; }
message ModelProto { int64 ir_version=1; GraphProto graph=2; }
`

function toTensorProto(name, tensor, quantize=false) {
  const arr = tensor.dataSync()
  let raw
  let dtype
  if (quantize) {
    const q = new Int8Array(arr.length)
    const max = Math.max(...arr.map(Math.abs)) || 1
    const scale = max/127
    for (let i=0;i<arr.length;i++) q[i] = Math.round(arr[i]/scale)
    raw = q
    dtype = 'INT8'
  } else {
    const f32 = new Float32Array(arr.length)
    for (let i=0;i<arr.length;i++) f32[i] = arr[i]
    raw = f32
    dtype = 'FLOAT'
  }
  return { name, dims: tensor.shape.map(s=>BigInt(s)), raw, dtype }
}

export async function exportOnnx(model, { quantize=false }={}) {
  const root = Root.fromJSON({})
  root.loadFromString(onnxProto)
  const ModelProto = root.lookupType('onnx.ModelProto')
  const GraphProto = root.lookupType('onnx.GraphProto')
  const NodeProto = root.lookupType('onnx.NodeProto')
  const TensorProto = root.lookupType('onnx.TensorProto')
  const ValueInfoProto = root.lookupType('onnx.ValueInfoProto')

  const nodes = []
  const inits = []
  const inputs = [{ name: 'input' }]
  const outputs = [{ name: 'output' }]

  let last = 'input'
  for (const layer of model.layers) {
    const conf = layer.getConfig()
    if (layer.name.includes('conv2d')) {
      const k = layer.getWeights()[0]
      const b = layer.getWeights()[1]
      const kName = layer.name+"_W"
      const bName = layer.name+"_B"
      const kProto = TensorProto.create({ data_type: TensorProto.DataType.FLOAT, dims: k.shape.map(BigInt), raw_data: new Uint8Array(new Float32Array(k.dataSync()).buffer) })
      const bProto = TensorProto.create({ data_type: TensorProto.DataType.FLOAT, dims: b.shape.map(BigInt), raw_data: new Uint8Array(new Float32Array(b.dataSync()).buffer) })
      inits.push(kProto, bProto)
      const out = layer.name+"_out"
      nodes.push(NodeProto.create({ op_type: 'Conv', input: [last, kName, bName], output: [out] }))
      last = out
    } else if (layer.name.includes('max_pooling2d')) {
      const out = layer.name+"_out"
      nodes.push(NodeProto.create({ op_type: 'MaxPool', input: [last], output: [out] }))
      last = out
    } else if (layer.name.includes('flatten')) {
      const out = layer.name+"_out"
      nodes.push(NodeProto.create({ op_type: 'Flatten', input: [last], output: [out] }))
      last = out
    } else if (layer.name.includes('dense')) {
      const w = layer.getWeights()[0]
      const b = layer.getWeights()[1]
      const wName = layer.name+"_W"
      const bName = layer.name+"_B"
      const wProto = TensorProto.create({ data_type: TensorProto.DataType.FLOAT, dims: w.shape.map(BigInt), raw_data: new Uint8Array(new Float32Array(w.dataSync()).buffer) })
      const bProto = TensorProto.create({ data_type: TensorProto.DataType.FLOAT, dims: b.shape.map(BigInt), raw_data: new Uint8Array(new Float32Array(b.dataSync()).buffer) })
      inits.push(wProto, bProto)
      const out = layer.name+"_out"
      nodes.push(NodeProto.create({ op_type: 'Gemm', input: [last, wName, bName], output: [out] }))
      last = out
    } else if (layer.name.includes('activation') || conf.activation === 'relu') {
      const out = layer.name+"_out"
      nodes.push(NodeProto.create({ op_type: 'Relu', input: [last], output: [out] }))
      last = out
    } else if (conf.activation === 'softmax') {
      const out = layer.name+"_out"
      nodes.push(NodeProto.create({ op_type: 'Softmax', input: [last], output: [out] }))
      last = out
    }
  }

  const graph = GraphProto.create({ name: 'cifar10', node: nodes, initializer: inits, input: inputs.map(i=>ValueInfoProto.create(i)), output: outputs.map(o=>ValueInfoProto.create(o)) })
  const modelProto = ModelProto.create({ ir_version: BigInt(8), graph })
  const buf = ModelProto.encode(modelProto).finish()
  return new Blob([buf], { type: 'application/octet-stream' })
}