import { render, screen } from '@testing-library/react'
import React from 'react'
import TrainingProgress from '../components/TrainingProgress.jsx'

it('renders canvases and stats', () => {
  const progress = { status: 'training', epoch: 2, batch: 3, timeMs: 1234, loss: [1,0.9,0.8], acc: [0.2,0.3,0.4] }
  render(<TrainingProgress progress={progress} />)
  expect(screen.getByLabelText('Loss graph')).toBeDefined()
  expect(screen.getByLabelText('Accuracy graph')).toBeDefined()
  expect(screen.getByText('Current epoch')).toBeDefined()
})