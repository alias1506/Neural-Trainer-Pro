import React, { useState } from 'react'

export default function Sidebar({ collapsed, onToggle, section, onNavigate }) {
  const [hoveredItem, setHoveredItem] = useState(null)
  
  const Item = ({ id, label, icon }) => (
    <div className="relative">
      <button 
        aria-label={label} 
        className={`flex items-center w-full py-2 rounded hover:bg-gray-200 focus-ring ${section===id?'bg-blue-100 text-blue-700 font-semibold':'text-gray-700'} ${collapsed ? 'justify-center px-0' : 'px-3'}`} 
        onClick={() => onNavigate(id)}
        onMouseEnter={() => setHoveredItem(id)}
        onMouseLeave={() => setHoveredItem(null)}
      >
        <span className={`text-xl ${collapsed ? '' : 'mr-3'}`} aria-hidden>{icon}</span>
        <span className={`transition-opacity duration-200 overflow-hidden whitespace-nowrap ${collapsed ? 'opacity-0 w-0' : 'opacity-100 w-auto'}`}>
          {label}
        </span>
      </button>
      {collapsed && hoveredItem === id && (
        <div className="absolute left-full top-0 ml-2 px-3 py-2 bg-gray-800 text-white text-sm rounded shadow-lg whitespace-nowrap z-50 animate-fade-in">
          {label}
        </div>
      )}
    </div>
  )
  return (
    <aside className={`h-screen sticky top-0 bg-gray-50 border-r border-gray-300 text-gray-800 ${collapsed?'w-16':'w-64'} transition-all duration-300`}>
      <div className={`p-3 flex items-center transition-all duration-300 ${collapsed ? 'justify-center' : 'justify-between'}`}>
        <h1 className={`font-semibold text-gray-800 transition-all duration-300 overflow-hidden whitespace-nowrap ${collapsed ? 'w-0 opacity-0' : 'w-auto opacity-100'}`}>
          NeuralTrainer Pro
        </h1>
        <button className="focus-ring p-2 rounded" aria-label="Toggle sidebar" onClick={onToggle}>
          {collapsed ? (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
              <path d="M9 6l6 6-6 6" />
            </svg>
          ) : (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
              <path d="M15 6l-6 6 6 6" />
            </svg>
          )}
        </button>
      </div>
      <nav className="p-3 space-y-1" role="navigation" aria-label="Main">
        <Item id="dataset" label="Dataset Selection" icon="📁" />
        <Item id="config" label="Training Configuration" icon="⚙️" />
        <Item id="progress" label="Training Progress" icon="📈" />
        <Item id="export" label="Model Export" icon="⬇️" />
        <Item id="history" label="Training History" icon="🕘" />
      </nav>
    </aside>
  )
}