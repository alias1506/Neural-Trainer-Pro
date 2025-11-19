import { render, screen, fireEvent } from '@testing-library/react'
import React from 'react'
import DatasetSelector from '../components/DatasetSelector.jsx'

function makeFile(name, content) {
  return new File([content], name, { type: 'application/octet-stream' })
}

it('accepts drag and drop files', async () => {
  const handler = vi.fn()
  render(<DatasetSelector onFilesSelected={handler} />)
  const dropzone = screen.getByRole('button', { name: /Drag and drop/i })
  const meta = makeFile('batches.meta.txt', 'airplane\nautomobile')
  const data = makeFile('data_batch_1.bin', new Uint8Array(3073))
  fireEvent.drop(dropzone, { dataTransfer: { files: [meta, data] } })
  expect(handler).toHaveBeenCalled()
})

it('removes selection when clicking remove', () => {
  const onFilesSelected = vi.fn()
  const onClear = vi.fn()
  render(<DatasetSelector onFilesSelected={onFilesSelected} onClear={onClear} selectedFiles={[makeFile('data_batch_1.bin', new Uint8Array(3073))]} />)
  fireEvent.click(screen.getByText('Remove Selection'))
  expect(onClear).toHaveBeenCalled()
})