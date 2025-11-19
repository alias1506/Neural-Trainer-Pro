import { render, screen, fireEvent } from '@testing-library/react'
import React from 'react'
import TrainingHistory from '../components/TrainingHistory.jsx'

it('renders history and loads config', () => {
  const history = [{ id: 1, date: new Date().toISOString(), config: { epochs: 5, batchSize: 16, learningRate: 0.01 }, metrics: { acc: 0.8, loss: 0.3 } }]
  const onLoad = vi.fn()
  render(<TrainingHistory history={history} onLoad={onLoad} />)
  fireEvent.click(screen.getByText('Load config'))
  expect(onLoad).toHaveBeenCalled()
})