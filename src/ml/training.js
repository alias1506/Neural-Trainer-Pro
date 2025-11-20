import * as tf from '@tensorflow/tfjs'

export async function trainModel(model, dataset, config, onEvent) {
  const numImages = dataset.labels.length
  
  // Create tensors and normalize
  const xs = tf.tidy(() => {
    const t = tf.tensor4d(Array.from(dataset.images), [numImages, 32, 32, 3], 'float32')
    return t.div(255.0)
  })
  const ys = tf.tidy(() => tf.oneHot(tf.tensor1d(dataset.labels, 'int32'), 10))
  
  const epochLoss = []
  const epochAcc = []
  
  const history = await model.fit(xs, ys, {
    epochs: config.epochs,
    batchSize: config.batchSize,
    shuffle: true,
    validationSplit: 0.1,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        epochLoss.push(logs.loss)
        epochAcc.push(logs.acc || logs.accuracy || 0)
        
        // Let browser update UI
        await new Promise(resolve => setTimeout(resolve, 0))
        
        onEvent({ 
          type: 'epoch', 
          epoch: epoch + 1, 
          loss: [...epochLoss], 
          acc: [...epochAcc] 
        })
      },
      onTrainEnd: () => {
        onEvent({ type: 'end' })
      }
    }
  })
  
  xs.dispose()
  ys.dispose()
  
  return { 
    history, 
    metrics: { 
      acc: epochAcc[epochAcc.length - 1] || 0, 
      loss: epochLoss[epochLoss.length - 1] || 0 
    }
  }
}