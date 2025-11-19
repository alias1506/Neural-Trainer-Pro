export function drawSeries(canvas, series, { color = '#3f8cff', label = '' } = {}) {
  if (!canvas) return
  const ctx = canvas.getContext && canvas.getContext('2d', { alpha: false })
  if (!ctx) return
  
  if (series.length === 0) {
    // Draw empty state
    const w = canvas.width, h = canvas.height
    ctx.clearRect(0,0,w,h)
    ctx.fillStyle = '#f9fafb'
    ctx.fillRect(0,0,w,h)
    ctx.fillStyle = '#9ca3af'
    ctx.font = '14px system-ui'
    ctx.textAlign = 'center'
    ctx.fillText('Waiting for data...', w/2, h/2)
    return
  }
  const w = canvas.width, h = canvas.height
  const padding = { left: 60, right: 20, top: 40, bottom: 40 }
  
  // Use willReadFrequently for better performance
  ctx.clearRect(0,0,w,h)
  ctx.fillStyle = '#ffffff'
  ctx.fillRect(0,0,w,h)
  
  // Calculate min/max for Y axis
  const maxY = Math.max(1e-6, ...series)
  const minY = Math.min(0, ...series)
  const rangeY = maxY - minY || 1
  
  // Draw grid lines (reduced for performance)
  ctx.strokeStyle = '#e5e7eb'
  ctx.lineWidth = 1
  ctx.beginPath()
  for (let i=0; i<=5; i++) {
    const y = padding.top + (h - padding.top - padding.bottom) * i / 5
    ctx.moveTo(padding.left, y)
    ctx.lineTo(w - padding.right, y)
  }
  ctx.stroke()
  
  // Draw Y axis labels
  ctx.fillStyle = '#6b7280'
  ctx.font = '11px system-ui'
  ctx.textAlign = 'right'
  for (let i=0; i<=5; i++) {
    const y = padding.top + (h - padding.top - padding.bottom) * i / 5
    const value = maxY - (rangeY * i / 5)
    ctx.fillText(value.toFixed(3), padding.left - 10, y + 4)
  }
  
  // Draw X axis labels (reduce number for performance)
  ctx.textAlign = 'center'
  const xLabels = Math.min(5, series.length)
  for (let i=0; i<xLabels; i++) {
    const x = padding.left + (w - padding.left - padding.right) * i / (xLabels-1 || 1)
    const index = Math.floor(series.length * i / (xLabels-1 || 1))
    ctx.fillText(index.toString(), x, h - padding.bottom + 20)
  }
  
  // Draw axis labels
  ctx.fillStyle = '#374151'
  ctx.font = 'bold 12px system-ui'
  ctx.textAlign = 'center'
  ctx.fillText('Iteration', w / 2, h - 5)
  
  ctx.save()
  ctx.translate(15, h / 2)
  ctx.rotate(-Math.PI / 2)
  ctx.fillText('Value', 0, 0)
  ctx.restore()
  
  // Draw series line - heavily downsample for performance
  ctx.strokeStyle = color
  ctx.lineWidth = 2
  ctx.beginPath()
  
  const maxPoints = 200 // Reduce max points for better performance
  const step = Math.max(1, Math.floor(series.length / maxPoints))
  
  const chartWidth = w - padding.left - padding.right
  const chartHeight = h - padding.top - padding.bottom
  
  for (let i = 0; i < series.length; i += step) {
    const y = series[i]
    const x = padding.left + (i/(series.length-1||1)) * chartWidth
    const yy = padding.top + chartHeight - ((y-minY)/rangeY) * chartHeight
    if (i===0) ctx.moveTo(x, yy); else ctx.lineTo(x, yy)
  }
  
  // Make sure to draw the last point
  if (series.length > 1 && (series.length - 1) % step !== 0) {
    const y = series[series.length - 1]
    const x = w - padding.right
    const yy = padding.top + chartHeight - ((y-minY)/rangeY) * chartHeight
    ctx.lineTo(x, yy)
  }
  
  ctx.stroke()
  
  // Draw label
  ctx.fillStyle = '#111827'
  ctx.font = 'bold 14px system-ui'
  ctx.textAlign = 'left'
  ctx.fillText(label, padding.left, 20)
}

export function drawCombinedChart(canvas, { loss, accuracy, epoch }) {
  if (!canvas) return
  const ctx = canvas.getContext('2d', { alpha: false })
  if (!ctx) return
  if (!loss || loss.length === 0) return
  
  const w = canvas.width, h = canvas.height
  const padding = { left: 80, right: 80, top: 60, bottom: 80 }
  
  ctx.fillStyle = '#ffffff'
  ctx.fillRect(0, 0, w, h)
  
  // Simple downsampling
  const maxPoints = 100
  const step = Math.max(1, Math.floor(loss.length / maxPoints))
  const sampledLoss = []
  const sampledAcc = []
  for (let i = 0; i < loss.length; i += step) {
    sampledLoss.push(loss[i])
    sampledAcc.push(accuracy[i])
  }
  
  // Calculate ranges
  const maxLoss = Math.max(...sampledLoss)
  const minLoss = Math.min(0, ...sampledLoss)
  const rangeLoss = maxLoss - minLoss || 1
  
  const maxAcc = 1 // Accuracy is 0-1
  const minAcc = 0
  const rangeAcc = maxAcc - minAcc
  
  const chartWidth = w - padding.left - padding.right
  const chartHeight = h - padding.top - padding.bottom
  
  // Draw grid
  ctx.strokeStyle = '#e5e7eb'
  ctx.lineWidth = 1
  ctx.beginPath()
  for (let i=0; i<=5; i++) {
    const y = padding.top + chartHeight * i / 5
    ctx.moveTo(padding.left, y)
    ctx.lineTo(w - padding.right, y)
  }
  ctx.stroke()
  
  // Draw Y axis labels (Loss - left side)
  ctx.fillStyle = '#e11d48'
  ctx.font = '11px system-ui'
  ctx.textAlign = 'right'
  for (let i=0; i<=5; i++) {
    const y = padding.top + chartHeight * i / 5
    const value = maxLoss - (rangeLoss * i / 5)
    ctx.fillText(value.toFixed(2), padding.left - 10, y + 4)
  }
  
  // Draw Y axis labels (Accuracy - right side)
  ctx.fillStyle = '#059669'
  ctx.textAlign = 'left'
  for (let i=0; i<=5; i++) {
    const y = padding.top + chartHeight * i / 5
    const value = maxAcc - (rangeAcc * i / 5)
    ctx.fillText((value * 100).toFixed(0) + '%', w - padding.right + 10, y + 4)
  }
  
  // Draw X axis labels (Epochs)
  ctx.fillStyle = '#6b7280'
  ctx.textAlign = 'center'
  const xLabels = Math.min(6, epoch || 1)
  for (let i=0; i<=xLabels; i++) {
    const x = padding.left + chartWidth * i / xLabels
    ctx.fillText(i.toString(), x, h - padding.bottom + 20)
  }
  
  // Draw axis titles
  ctx.fillStyle = '#374151'
  ctx.font = 'bold 12px system-ui'
  ctx.fillText('Epoch', w / 2, h - 10)
  
  ctx.save()
  ctx.translate(20, h / 2)
  ctx.rotate(-Math.PI / 2)
  ctx.fillStyle = '#e11d48'
  ctx.fillText('Loss', 0, 0)
  ctx.restore()
  
  ctx.save()
  ctx.translate(w - 20, h / 2)
  ctx.rotate(Math.PI / 2)
  ctx.fillStyle = '#059669'
  ctx.fillText('Accuracy', 0, 0)
  ctx.restore()
  
  // Draw Loss line
  ctx.strokeStyle = '#e11d48'
  ctx.lineWidth = 2
  ctx.beginPath()
  for (let i = 0; i < sampledLoss.length; i++) {
    const x = padding.left + (i/(sampledLoss.length-1||1)) * chartWidth
    const y = padding.top + chartHeight - ((sampledLoss[i]-minLoss)/rangeLoss) * chartHeight
    if (i===0) ctx.moveTo(x, y); else ctx.lineTo(x, y)
  }
  ctx.stroke()
  
  // Draw Accuracy line
  ctx.strokeStyle = '#059669'
  ctx.lineWidth = 2
  ctx.beginPath()
  for (let i = 0; i < sampledAcc.length; i++) {
    const x = padding.left + (i/(sampledAcc.length-1||1)) * chartWidth
    const y = padding.top + chartHeight - ((sampledAcc[i]-minAcc)/rangeAcc) * chartHeight
    if (i===0) ctx.moveTo(x, y); else ctx.lineTo(x, y)
  }
  ctx.stroke()
  
  // Title
  ctx.fillStyle = '#111827'
  ctx.font = 'bold 16px system-ui'
  ctx.textAlign = 'center'
  ctx.fillText('Training Metrics', w / 2, 30)
  
  // Draw Legend at bottom
  const legendY = h - 30
  const legendSpacing = 150
  const legendStartX = w / 2 - legendSpacing / 2
  
  // Loss legend
  ctx.fillStyle = '#e11d48'
  ctx.fillRect(legendStartX - 60, legendY - 3, 30, 3)
  ctx.fillStyle = '#374151'
  ctx.font = '13px system-ui'
  ctx.textAlign = 'left'
  ctx.fillText('Loss', legendStartX - 25, legendY + 2)
  
  // Accuracy legend
  ctx.fillStyle = '#059669'
  ctx.fillRect(legendStartX + 50, legendY - 3, 30, 3)
  ctx.fillStyle = '#374151'
  ctx.fillText('Accuracy', legendStartX + 85, legendY + 2)
}