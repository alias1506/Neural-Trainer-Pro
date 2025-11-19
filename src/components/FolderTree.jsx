import React, { useState } from 'react'

const FolderTreeNode = ({ name, path, isFolder, children, level = 0 }) => {
  // Auto-expand only the root folder (level 0), keep all subfolders collapsed
  const [expanded, setExpanded] = useState(level === 0)

  const paddingLeft = level * 20 // Increased spacing for better visual hierarchy

  // File rendering
  if (!isFolder) {
    return (
      <div 
        className="flex items-center py-1 px-2 hover:bg-gray-50 rounded text-sm"
        style={{ paddingLeft: `${paddingLeft + 24}px` }}
        title={name} // Show full name on hover
      >
        <svg className="w-4 h-4 mr-2 text-blue-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clipRule="evenodd"/>
        </svg>
        <span className="text-gray-700 truncate">{name}</span>
      </div>
    )
  }

  // Folder rendering with recursive children support
  const hasChildren = children && children.length > 0
  const folderCount = hasChildren ? children.filter(c => c.isFolder).length : 0
  const fileCount = hasChildren ? children.filter(c => !c.isFolder).length : 0

  return (
    <div>
      <div 
        className="flex items-center py-1 px-2 hover:bg-gray-100 rounded cursor-pointer text-sm group select-none"
        style={{ paddingLeft: `${paddingLeft}px` }}
        onClick={() => hasChildren && setExpanded(!expanded)}
        title={hasChildren ? `Click to ${expanded ? 'collapse' : 'expand'}` : 'Empty folder'}
      >
        {hasChildren ? (
          <svg 
            className={`w-4 h-4 mr-2 text-gray-500 flex-shrink-0 transition-transform duration-200 ${expanded ? 'rotate-90' : ''}`} 
            fill="currentColor" 
            viewBox="0 0 20 20"
          >
            <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd"/>
          </svg>
        ) : (
          <span className="w-4 h-4 mr-2 flex-shrink-0" />
        )}
        <svg 
          className={`w-4 h-4 mr-2 flex-shrink-0 transition-colors ${expanded ? 'text-yellow-600' : 'text-yellow-500'}`} 
          fill="currentColor" 
          viewBox="0 0 20 20"
        >
          {expanded ? (
            <path d="M2 6a2 2 0 012-2h5l2 2h5a2 2 0 012 2v6a2 2 0 01-2 2H4a2 2 0 01-2-2V6z"/>
          ) : (
            <path fillRule="evenodd" d="M2 6a2 2 0 012-2h4l3 3h5a2 2 0 012 2v6a2 2 0 01-2 2H4a2 2 0 01-2-2V6z" clipRule="evenodd"/>
          )}
        </svg>
        <span className="font-medium text-gray-800 truncate">{name}</span>
        {hasChildren && (
          <span className="ml-auto pl-2 text-xs text-gray-500 flex-shrink-0 whitespace-nowrap">
            {fileCount > 0 && `${fileCount} ${fileCount === 1 ? 'file' : 'files'}`}
            {fileCount > 0 && folderCount > 0 && ', '}
            {folderCount > 0 && `${folderCount} ${folderCount === 1 ? 'folder' : 'folders'}`}
          </span>
        )}
      </div>
      {expanded && hasChildren && (
        <div>
          {/* Render folders first, then files for better organization */}
          {children
            .sort((a, b) => {
              // Folders before files
              if (a.isFolder && !b.isFolder) return -1
              if (!a.isFolder && b.isFolder) return 1
              // Alphabetical within same type
              return a.name.localeCompare(b.name)
            })
            .map((child, idx) => (
              <FolderTreeNode 
                key={`${child.path}-${idx}`} 
                name={child.name} 
                path={child.path}
                isFolder={child.isFolder}
                children={child.children}
                level={level + 1}
              />
            ))}
        </div>
      )}
    </div>
  )
}

// Count total files recursively
const countFiles = (children) => {
  if (!children) return 0
  let count = 0
  for (const child of children) {
    if (child.isFolder) {
      count += countFiles(child.children)
    } else {
      count++
    }
  }
  return count
}

export default function FolderTree({ structure }) {
  if (!structure || !structure.tree) {
    return (
      <div className="text-sm text-gray-500 p-4 text-center">
        No folder structure available
      </div>
    )
  }

  const totalFiles = countFiles(structure.tree.children)

  return (
    <div className="bg-white rounded-lg border border-gray-200">
      <div className="p-3 max-h-[600px] overflow-y-auto custom-scrollbar">
        <FolderTreeNode 
          name={structure.tree.name || 'Dataset'} 
          path={structure.tree.path || ''}
          isFolder={true}
          children={structure.tree.children}
          level={0}
        />
        {totalFiles > 0 && (
          <div className="mt-2 pt-2 border-t border-gray-200 text-xs text-gray-500 text-center">
            Total files: {totalFiles}
          </div>
        )}
      </div>
    </div>
  )
}
