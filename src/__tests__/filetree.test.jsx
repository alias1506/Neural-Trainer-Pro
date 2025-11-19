import { render, screen, fireEvent } from '@testing-library/react'
import React from 'react'
import FileTree from '../components/FileTree.jsx'

it('toggles folder open/close and shows files', () => {
  const items = [{ name: 'dataset', type: 'dir', children: [{ name: 'data_batch_1.bin', type: 'file' }] }]
  render(<FileTree items={items} />)
  expect(screen.getByText('dataset')).toBeDefined()
  expect(screen.getByText('data_batch_1.bin')).toBeDefined()
  const btn = screen.getByLabelText('Toggle folder')
  fireEvent.click(btn)
})