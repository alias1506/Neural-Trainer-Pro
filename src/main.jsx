import React from 'react'
import { createRoot } from 'react-dom/client'
import 'sweetalert2/dist/sweetalert2.min.css'
import './styles/index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(<App />)