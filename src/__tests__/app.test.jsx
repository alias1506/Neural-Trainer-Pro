import { render, screen, fireEvent } from '@testing-library/react'
import React from 'react'
import App from '../App.jsx'

describe('App UI', () => {
  it('renders sidebar sections', () => {
    render(<App />)
    const ds = screen.getByRole('button', { name: 'Dataset Selection' })
    const cfg = screen.getByRole('button', { name: 'Training Configuration' })
    const exp = screen.getByRole('button', { name: 'Model Export' })
    expect(ds).toBeDefined()
    expect(cfg).toBeDefined()
    expect(exp).toBeDefined()
  })
  it('shows config summary', () => {
    render(<App />)
    fireEvent.click(screen.getByLabelText('Training Configuration'))
    expect(screen.getByText(/Configuration Summary/)).toBeDefined()
  })
})