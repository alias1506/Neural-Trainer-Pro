import { render, screen } from '@testing-library/react'
import React from 'react'
import ModelExport from '../components/ModelExport.jsx'

it('shows export buttons', () => {
  render(<ModelExport model={{ save: async ()=>{} }} trained={{ metrics: { acc: 0.9, loss: 0.1 }, elapsedMs: 1000 }} />)
  expect(screen.getByText('Download .tfjs')).toBeDefined()
  expect(screen.getByText('Download .onnx')).toBeDefined()
})