export async function parseCifarBatches(files, onProgress = ()=>{}) {
  const dataFiles = files.filter(f => /data_batch_\d+\.bin|test_batch\.bin/.test(f.name)).sort((a,b)=>a.name.localeCompare(b.name))
  
  // Limit to first file only for faster loading
  const limitedFiles = dataFiles.slice(0, 1)
  
  const labels = []
  const images = []
  
  // Process files
  for (let i=0;i<limitedFiles.length;i++) {
    const buf = await limitedFiles[i].arrayBuffer()
    const view = new Uint8Array(buf)
    const record = 3073
    const count = Math.floor(view.length / record)
    
    for (let r=0;r<count;r++) {
      const base = r*record
      labels.push(view[base])
      
      // Copy image data directly
      const imgData = new Uint8Array(3072)
      let idx = 0
      for (let p=0;p<1024;p++) {
        imgData[idx++] = view[base+1+p]        // R
        imgData[idx++] = view[base+1+1024+p]   // G
        imgData[idx++] = view[base+1+2048+p]   // B
      }
      images.push(...imgData)
    }
    onProgress({ fileIndex: i+1, total: limitedFiles.length })
  }
  
  return { images: new Uint8Array(images), labels }
}