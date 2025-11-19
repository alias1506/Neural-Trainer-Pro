import React, { useState } from 'react'

export default function FileTree({ items }) {
  return (
    <ul className="text-sm text-gray-700">
      {items.map((item, idx) => (
        <TreeItem key={idx} item={item} depth={0} />
      ))}
    </ul>
  )
}

function TreeItem({ item, depth }) {
  const [open, setOpen] = useState(true)
  const isDir = item.type === 'dir'
  return (
    <li className="py-1">
      <div className="flex items-center" style={{ paddingLeft: depth*12 }}>
        {isDir ? (
          <button className="mr-2 p-1 rounded focus-ring" aria-label="Toggle folder" onClick={()=>setOpen(!open)}>
            {open ? (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden><path d="M15 6l-6 6 6 6" /></svg>
            ) : (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden><path d="M9 6l6 6-6 6" /></svg>
            )}
          </button>
        ) : (
          <span className="mr-2" aria-hidden>📄</span>
        )}
        <span className="truncate">{item.name}</span>
      </div>
      {isDir && open && item.children?.length>0 && (
        <ul>
          {item.children.map((c, i) => (
            <TreeItem key={i} item={c} depth={depth+1} />
          ))}
        </ul>
      )}
    </li>
  )
}