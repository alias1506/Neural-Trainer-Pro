import * as tf from '@tensorflow/tfjs'

export function createModel(config) {
  const model = tf.sequential()
  
  // Simple CNN
  model.add(tf.layers.conv2d({ 
    inputShape: [32,32,3], 
    filters: 32, 
    kernelSize: 3, 
    activation: 'relu'
  }))
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }))
  
  model.add(tf.layers.conv2d({ 
    filters: 64, 
    kernelSize: 3, 
    activation: 'relu'
  }))
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }))
  
  model.add(tf.layers.flatten())
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }))
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' }))
  
  // Use Adam optimizer
  const opt = config.optimizer==='adam' 
    ? tf.train.adam(config.learningRate) 
    : config.optimizer==='sgd' 
    ? tf.train.sgd(config.learningRate, 0.9)
    : tf.train.rmsprop(config.learningRate)
    
  model.compile({ 
    optimizer: opt, 
    loss: 'categoricalCrossentropy', 
    metrics: ['accuracy'] 
  })
  
  return model
}