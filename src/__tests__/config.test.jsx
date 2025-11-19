import { render, screen, fireEvent } from '@testing-library/react'
import React from 'react'
import TrainingConfig from '../components/TrainingConfig.jsx'

it('renders defaults and summary', () => {
  const cfg = { epochs: 50, batchSize: 32, learningRate: 0.001, optimizer: 'adam' }
  const onChange = vi.fn()
  const onStart = vi.fn()
  render(<TrainingConfig config={cfg} onChange={onChange} onStart={onStart} datasetReady={true} />)
  expect(screen.getByText(/Configuration Summary/)).toBeDefined()
  fireEvent.click(screen.getByText('Start Training'))
  expect(onStart).toHaveBeenCalled()
})