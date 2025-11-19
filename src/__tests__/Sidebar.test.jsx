import { render, screen, fireEvent } from '@testing-library/react'
import React from 'react'
import Sidebar from '../components/Sidebar.jsx'

it('toggles collapse', () => {
  const onToggle = vi.fn()
  render(<Sidebar collapsed={false} onToggle={onToggle} section={'dataset'} onNavigate={()=>{}} />)
  fireEvent.click(screen.getByLabelText('Toggle sidebar'))
  expect(onToggle).toHaveBeenCalled()
})